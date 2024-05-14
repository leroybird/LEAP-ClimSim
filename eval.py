import polars as pl
import tqdm
import train
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse
import config
import norm
import pytorch_lightning as L


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


def get_predictions(model, test_loader):
    trainer = L.Trainer(
        precision=16,
    )
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test_df", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--bs", type=int, default=128)

    args = parser.parse_args()

    test_df = pl.read_parquet(args.test_df)

    cfg_loader = config.LoaderConfig()
    cfg_data = config.get_data_config(cfg_loader)

    lit_model = train.get_model(cfg_data, cfg_loader, args.model)

    x_test = test_df.select(cfg_data.x_names).to_numpy()
    norm_x, norm_y = norm.get_stats(cfg_loader, cfg_data)

    test_ds = EvalLoader({"x": x_test}, {"x": norm_x})
    test_loader = DataLoader(test_ds, batch_size=args.bs, drop_last=False, shuffle=False, num_workers=0, pin_memory=True)

    preds = get_predictions(lit_model, test_loader)
    print(preds)
    save_predictions(preds, args.output, test_df, norm_y, cfg_data.y_names)
