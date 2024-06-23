from collections import defaultdict
from functools import cache
import gc
from pathlib import Path
from turtle import setup
import polars as pl
import tqdm
import train
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse
import config
import norm
import lightning as L
import xarray as xr
from dataloader import get_static, get_datasets


class EvalLoader(Dataset):
    def __init__(self, data_dict, norm_dict):
        self.data_dict = data_dict
        self.norm_dict = norm_dict

    def __getitem__(self, idx):
        x = self.data_dict["x"][idx].copy()
        y = self.data_dict["y"][idx].copy() if "y" in self.data_dict else np.zeros(1)
        x = self.norm_dict["x"](x[None, :])[0]
        y = self.norm_dict["y"](y[None, :])[0] if "y" in self.data_dict else np.zeros(1)
        return x.astype(np.float32)

    def __len__(self):
        return self.data_dict["x"].shape[0]


@cache
def get_static_ds(path):
    with xr.open_dataset(path) as ds:
        static_ds = ds.load()
    return get_static(static_ds)


@cache
def get_static_idx(idx, path):
    static_data = get_static_ds(path)
    assert idx < len(static_data)
    return static_data[idx, :]


class EvalLoaderTime(Dataset):
    def __init__(
        self,
        data_dict,
        norm_dict,
        num_grid_cells=384,
        use_static: bool = True,
        add_time: bool = True,
        static_path: str = "/mnt/storage/kaggle/ClimSim_low-res_grid-info.nc",
        neighbours_path: Path = Path(__file__).parent / "neighbours.nc",
    ):

        self.data_dict = data_dict
        self.norm_dict = norm_dict
        self.num_grid_cells = num_grid_cells
        self.use_static = use_static
        self.add_time = add_time
        self.offset = self.num_grid_cells if self.add_time else 0
        self.static_path = static_path

        self.neighbours = xr.open_dataset(neighbours_path).load()
        self.neighbours.close()

    def get_data(self, idx):
        x = self.data_dict["x"][idx].copy()
        y = self.data_dict["y"][idx].copy() if "y" in self.data_dict else np.zeros(1)
        if self.use_static:
            static_idx = idx % self.num_grid_cells
            x_static = get_static_idx(static_idx, self.static_path)
            x = np.concatenate([x, x_static])

        x = self.norm_dict["x"](x[None, :])[0]
        y = self.norm_dict["y"](y[None, :])[0] if "y" in self.data_dict else np.zeros(1)
        return x, y

    def get_all_neighbours(self, idx, t_val):
        idx_mod = idx % self.num_grid_cells
        idx_rel = idx - idx_mod

        x_cen, y = self.get_data(idx)
        x_nei = np.stack(
            [
                self.get_data(idx_rel + i)[0]
                for i in self.neighbours["idxs"][idx_mod].values
            ],
            axis=0,
        )

        assert (x_nei[0] == x_cen).all()

        # Set the ranges between -1 and 1
        y_dist = self.neighbours["y_distances"].values[idx_mod, :, None] / 1975267.0
        x_dist = self.neighbours["x_distances"].values[idx_mod, :, None] / 1975267.0
        dist_norm = self.neighbours["distances"].values[idx_mod, :, None] / 2035.0

        time_emb = np.ones_like(dist_norm) * t_val

        x_nei = np.concatenate(
            [x_nei, y_dist, x_dist, dist_norm, time_emb], axis=-1
        ).astype(np.float32)

        return x_nei, y

    def __getitem__(self, l_idx):
        c_idx = l_idx + self.offset

        x0, _ = self.get_all_neighbours(l_idx, -1.0)
        x1, y = self.get_all_neighbours(c_idx, 0.0)
        x2, _ = self.get_all_neighbours(c_idx + self.num_grid_cells, 1.0)
        return np.concatenate([x1, x0, x2], axis=0).astype(np.float32)

        # return (
        #     np.concatenate([x1, x0, x2], axis=0).astype(np.float32),
        #     y,
        # )  # np.stack([x0, x1, x2], axis=-1).astype(np.float32)

    def __len__(self):

        total = self.data_dict["x"].shape[0]
        rem = total % self.num_grid_cells
        return total - rem - 2 * self.num_grid_cells


def get_predictions(model, test_loader):
    trainer = L.Trainer()
    preds = trainer.predict(
        model,
        test_loader,
    )
    preds = np.concatenate([p.numpy().astype(np.float32) for p in preds])
    print(preds.shape)
    return preds


def save_predictions(preds, output_path, test_df, norm_y, y_names):
    output = pl.DataFrame(test_df["sample_id"])
    # We don't need to denorm y as we predict in the data distribution
    p_cols = []
    for n, col in enumerate(y_names):
        if norm_y.zero_mask[n]:
            print(f"skipping {col}")
            pl_col = pl.lit(0.0, dtype=pl.Float32).alias(col)
        else:
            pl_col = pl.Series(col, preds[:, n], dtype=pl.Float32)
        p_cols.append(pl_col)

    output = output.with_columns(p_cols)

    print("Saving to:", output_path)
    output.write_csv(output_path)


def zero_cols(preds, norm_y):
    for i, zeroed in enumerate(norm_y.zero_mask):
        if zeroed:
            # print(f"Zeroing column {i}")
            preds[:, i] = 0.0
    return preds


def save_parquet(preds, output_path):
    df = pl.DataFrame(preds)
    df.write_parquet(output_path)


class CombinedDataset(Dataset):
    def __init__(self, ds1, ds2):
        self.ds1 = ds1
        self.ds2 = ds2
        assert len(ds1) == len(ds2)

    def __getitem__(self, idx):
        return idx, self.ds1[idx], self.ds2[idx]

    def __len__(self):
        return len(self.ds1)


def predict_save_train(
    model, ds_norm, ds_raw, max_steps, output_path, norm_y, shuffle=True
):
    model.eval()
    # Iterate over both datasets at the same time
    # idxs = np.random.choice(len(ds_norm), max_steps, replace=False)

    ds_stack = CombinedDataset(ds_norm, ds_raw)
    ds_loader = DataLoader(
        ds_stack,
        batch_size=1,
        drop_last=False,
        num_workers=16,
        pin_memory=False,
        shuffle=shuffle,
        collate_fn=lambda x: x[0],
    )
    output = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm.tqdm(enumerate(ds_loader), total=max_steps):
            n, (idx, (x_n, y_n), (x_r, y_r)) = batch

            preds = model(torch.from_numpy(x_n).cuda())
            preds = preds.cpu().detach().numpy()

            preds = zero_cols(preds, norm_y)
            # error = np.abs(preds - y_n).mean()

            # show error on the progress bar
            # tqdm.tqdm.write(f"Error: {error:.4f}")

            out_p = output_path / "pred_all.parquet"  # f"{str(n).zfill(5)}_pred.parquet"
            output[out_p].append(preds)
            # save_parquet(preds, out_p)

            out_x = output_path / "x_all.parquet"  # f"{str(n).zfill(5)}_x.parquet"
            output[out_x].append(x_r[:, 0])
            # Save only the centre grid cell
            # save_parquet(x_r[:, 0], out_x)

            out_y = output_path / "y_all.parquet"  # f"{str(n).zfill(5)}_y.parquet"
            output[out_y].append(y_r)

            output[output_path / "index.parquet"].append(np.array([idx] * 384))
            # save_parquet(y_r, out_y)

            if n == max_steps - 1:
                break

    out_keys = list(output.keys())
    for k in out_keys:
        data = output.pop(k)
        data = np.concatenate(data)
        gc.collect()

        print(f"Saving {k}")
        save_parquet(data, k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test_df", type=str, required=False)
    parser.add_argument(
        "--ds_type", type=str, default="test", choices=["test", "train", "valid"]
    )
    parser.add_argument("--train_ds_samples", type=int, default=10_000)

    parser.add_argument("--output", type=str, required=False)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--no_static", action="store_true")
    parser.add_argument("--no_time", action="store_true")
    parser.add_argument("--preds_only", action="store_true")

    args = parser.parse_args()

    cfg_loader = config.LoaderConfig()
    cfg_data = config.get_data_config(cfg_loader)
    cfg_loader.use_iterable_train = False

    lit_model = train.get_model(cfg_data, cfg_loader, args.model, setup_dataloader=False)

    norm_x, norm_y = norm.get_stats(cfg_loader, cfg_data)
    if args.ds_type == "test":
        test_df = pl.read_parquet(args.test_df)
        x_test = test_df.select(cfg_data.x_names).to_numpy()

        if args.no_time:
            assert args.no_static
            test_ds = EvalLoader({"x": x_test}, {"x": norm_x})
        else:
            test_ds = EvalLoaderTime({"x": x_test}, {"x": norm_x})

        test_loader = DataLoader(
            test_ds,
            batch_size=args.bs,
            drop_last=False,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        preds = get_predictions(lit_model, test_loader)
        preds = zero_cols(preds, norm_y)

        if args.preds_only:

            torch.save(preds, args.output)
        else:
            save_predictions(preds, args.output, test_df, norm_y, cfg_data.y_names)
    else:

        output_dir = args.output
        assert output_dir is not None
        output_dir = Path(output_dir)
        # assert output_dir.is_dir()
        output_dir.mkdir(exist_ok=True, parents=True)

        assert cfg_loader.apply_norm
        train_ds_norm, val_ds_norm = get_datasets(cfg_loader, cfg_data)
        cfg_loader.apply_norm = False

        train_ds_raw, val_ds_raw = get_datasets(cfg_loader, cfg_data)

        if args.ds_type == "train":
            train_ds = train_ds_raw
            train_norm = train_ds_norm
            num_samples = args.train_ds_samples
        elif args.ds_type == "valid":
            train_ds = val_ds_raw
            train_norm = val_ds_norm
            num_samples = min(len(val_ds_norm), args.train_ds_samples)
        else:
            raise ValueError("Invalid ds_type")

        predict_save_train(
            lit_model,
            train_norm,
            train_ds,
            num_samples,
            output_dir,
            norm_y,
        )
