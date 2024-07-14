from pydantic import BaseModel
import polars as pl


class LoaderConfig(BaseModel):
    base_path: str = "/data"
    index_path: str = f"{base_path}/index.parquet"
    grid_info_path: str = f"{base_path}/ClimSim_low-res_grid-info.nc"
    root_folder: str = f"{base_path}/trian"

    weights_path: str = f"{base_path}/sample_submission.csv"
    sample_submission_path: str = f"{base_path}/new_data/sample_submission.csv"
    train_kaggle_path: str = f"{base_path}/train_header.parquet"
    train_kaggle_csv: str = f"{base_path}/train.csv"
    test_kaggle_path: str = f"{base_path}/new_data/test.csv"
    x_stats_path: str = "x_stats_v2_1.json"
    y_stats_path: str = "y_stats_v2_1.json"

    num_workers: int = 24
    seed: int = 42
    sample_size: int = 16
    use_iterable_train: bool = True

    apply_norm: bool = True
    batch_size: int = 128

    x_tanh: bool = True
    x_mask_thresh: float | None = None

class DataConfig(BaseModel):
    num_vert: int = 60
    num_vert_feat: int = 9
    num_vert_feat_y: int = 6
    split_index: int = num_vert_feat_y * num_vert

    num_2d_feat: int = 16
    num_2d_feat_y: int = 8

    y_names: list[str] | None = None
    x_names: list[str] | None = None

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
