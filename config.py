from pydantic import BaseModel
import polars as pl


class LoaderConfig(BaseModel):
    index_path: str = "/mnt/ssd/kaggle/index_fwd.parquet"
    grid_info_path: str = "/mnt/storage/kaggle/ClimSim_low-res_grid-info.nc"
    root_folder: str = "/mnt/ssd/kaggle/train"

    weights_path: str = "/mnt/ssd/kaggle/sample_submission.csv"
    train_kaggle_path: str = "/mnt/ssd/kaggle/train2.parquet"

    x_stats_path: str = "x_stats8.json"

    num_workers: int = 24
    seed: int = 42
    sample_size: int = 16
    use_iterable_train: bool = True

    apply_norm: bool = True

    batch_size: int = 128


class DataConfig(BaseModel):
    num_vert: int = 60
    num_vert_feat: int = 9
    num_vert_feat_y: int = 6

    num_2d_feat: int = 16
    num_2d_feat_y: int = 8

    y_names: list[str] = None
    x_names: list[str] = None

    fac_idxs: tuple[int, int] = (num_vert, num_vert * 4)


def get_data_config(loader_cfg: LoaderConfig):
    pl.Config(tbl_cols=-1)
    train_df = pl.read_parquet(loader_cfg.train_kaggle_path, n_rows=1)

    x_names = train_df.columns[1:557]
    y_names = train_df.columns[557:]

    return DataConfig(y_names=y_names, x_names=x_names)


if __name__ == "__main__":
    loader_cfg = LoaderConfig()
    data_cfg = get_data_config(loader_cfg)
    print(data_cfg)
