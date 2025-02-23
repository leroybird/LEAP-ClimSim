from collections import defaultdict
from functools import partial
from pathlib import Path
from einops import rearrange
from lightning.pytorch.strategies import DDPStrategy
import argparse
from matplotlib import pyplot as plt
import numpy as np
import torch
import lightning as L
import torch.distributed
import wandb
import fastai.vision.all as fv
import torch.nn as nn
import yaml

import config
import arch
import dataloader

# import torch_optimizer as optim
from norm import get_classification_mask, load_from_json, get_stats

from schedulefree import AdamWScheduleFree
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.callbacks import ModelSummary, ModelCheckpoint
from lightning.pytorch.tuner import Tuner
import torch_optimizer as optim

# from kornia import losses

import robust_loss_pytorch.general

torch._dynamo.config.cache_size_limit = 512

# Enable tf32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def r_squared(pred, tar, mask=None, s_total=0.886):
    # s_total pre-calculated

    # if mask is not None:
    #     tar = tar[:, mask]
    #     pred = pred[:, mask]
    return 1 - (torch.mean((tar - pred) ** 2) / s_total)


# define the LightningModule
def masked_mse(pred, tar, mask):
    pred = pred[:, mask]
    tar = tar[:, mask]
    return torch.mean((pred - tar) ** 2)


def mse_t(pred, tar):
    mask = torch.zeros(tar.shape[1], dtype=torch.bool).to(tar.device)
    mask[0:60] = 1
    return masked_mse(pred, tar, mask)


def mse_q1(pred, tar):
    mask = torch.zeros(tar.shape[1], dtype=torch.bool).to(tar.device)
    mask[60:120] = 1
    return masked_mse(pred, tar, mask)


def mse_q2(pred, tar):
    mask = torch.zeros(tar.shape[1], dtype=torch.bool).to(tar.device)
    mask[120:180] = 1
    return masked_mse(pred, tar, mask)


def mse_q3(pred, tar):
    mask = torch.zeros(tar.shape[1], dtype=torch.bool).to(tar.device)
    mask[180:240] = 1
    return masked_mse(pred, tar, mask)


def mse_u(pred, tar):
    mask = torch.zeros(tar.shape[1], dtype=torch.bool).to(tar.device)
    mask[240:300] = 1
    return masked_mse(pred, tar, mask)


def mse_v(pred, tar):
    mask = torch.zeros(tar.shape[1], dtype=torch.bool).to(tar.device)
    mask[300:360] = 1
    return masked_mse(pred, tar, mask)


def mse_point(pred, tar):
    mask = torch.zeros(tar.shape[1], dtype=torch.bool).to(tar.device)
    mask[360:] = 1
    return masked_mse(pred, tar, mask)


class RegRatioClassLoss(nn.Module):
    def __init__(
        self,
        w_class=1.0,
        w_reg=1.0,
    ):
        super().__init__()
        self.reg_loss = nn.HuberLoss(delta=2.0, reduction="none")
        self.class_loss = nn.CrossEntropyLoss(reduction="none")

        self.w_class = w_class
        self.w_reg = w_reg

    def forward(self, pred: dict, batch: dict):
        reg_loss = self.reg_loss(pred["reg"], batch["y"])

        logit_class = pred["logits"]

        logit_class = rearrange(logit_class, "b c i -> (b c) i")
        target_class = rearrange(batch["y_cls"], "b c -> (b c)")

        class_loss = self.class_loss(logit_class, target_class)

        total = class_loss.mean() * self.w_class + reg_loss.mean() * self.w_reg

        total = torch.mean(total)
        return {
            "loss": total,
            "class_loss": torch.mean(class_loss),
            "reg_loss": torch.mean(reg_loss),
            "class_acc": (logit_class.argmax(dim=-1) == target_class).float().mean(),
        }


def correct_preds_cls(pred_batch: dict, targ_batch, y_norm):
    output = {}
    with torch.no_grad():
        mask_class_cols = y_norm.class_mask.squeeze()

        targ = targ_batch["y"].detach().cpu().numpy()
        targ_sub = targ[:, mask_class_cols]

        raw_reg = pred_batch["reg"].detach().cpu().numpy().copy()
        raw_reg_sub = raw_reg[:, mask_class_cols]

        def get_resi(pred, mask):
            r = raw_reg_sub.copy()
            r[mask] = pred[mask]
            return targ_sub - r

        def get_diff(pred, mask):
            return raw_reg[:, mask_class_cols][mask] - pred[:, mask_class_cols][mask]

        y_raw = targ_batch["y_raw"].detach().cpu().numpy()
        x_raw = targ_batch["x_raw"].detach().cpu().numpy()[:, 0 : y_raw.shape[1]]

        y_raw = y_raw[:, mask_class_cols]
        x_raw = x_raw[:, mask_class_cols]

        y_class = pred_batch["logits"].detach().cpu().numpy()
        y_class = np.argmax(y_class, axis=-1)

        assert y_class.shape == y_raw.shape

        mask_zero = y_class == 0
        mask_one = y_class == 1

        if mask_zero.sum() > 0:
            output["base"] = get_resi(raw_reg_sub, mask_zero)

        if mask_one.sum() > 0:
            raw_out = np.zeros_like(raw_reg_sub)
            raw_out[mask_one] = -x_raw[mask_one] / 1200
            assert not (raw_out == 0).all()
            raw_out -= y_norm.y_norm.means[:, mask_class_cols]
            raw_out = raw_out / y_norm.y_norm.stds[:, mask_class_cols]

            output["resi_neg"] = get_resi(raw_out, mask_one)
            # output["diff_neg"] = get_diff(raw_out, mask_one)

    return output


class LitModel(L.LightningModule):
    def __init__(
        self,
        model,
        cfg_data,
        cfg_loader,
        setup_dataloader=True,
        pt_compile=False,
        use_schedulefree=True,
    ):
        super().__init__()

        if pt_compile:
            model = torch.compile(model)
        self.model = model

        self.cfg_data = cfg_data
        self.cfg_loader = cfg_loader

        if setup_dataloader:
            self.train_loader, self.valid_loader = dataloader.setup_dataloaders(
                cfg_loader, cfg_data
            )

        self.x_norm, self.y_norm = get_stats(cfg_loader, cfg_data)

        self.y_class = cfg_loader.y_class
        if self.y_class:
            self.loss_func = RegRatioClassLoss()

            self.val_metrics = []
            self.train_metrics = []
        else:
            self.loss_func = nn.HuberLoss(delta=2.0)
            # self.loss_func = partial(losses.cauchy_loss, reduction='mean')

            self.val_metrics = [
                fv.mae,
                mse_t,
                mse_q1,
                mse_q2,
                mse_q3,
                mse_u,
                mse_v,
                mse_point,
            ]
            self.train_metrics = [
                fv.mse,
                mse_t,
                mse_q1,
                mse_q2,
                mse_q3,
                mse_u,
                mse_v,
                mse_point,
            ]

        self.learning_rate = 3e-4
        self.scheduler_steps = 200_000
        self.use_schedulefree = use_schedulefree
        self.mask = torch.zeros(360 + 8, dtype=torch.bool)
        self.mask[:] = True

        self.residuals = defaultdict(list)
        # self.mask[0:60] = True
        # self.mask[240:] = True

    def forward(self, x):
        return self.model(x)

    def step(self, batch, metrics=[], step_name="train", batch_idx=0):

        pred_batch = self.model(batch)
        pred = pred_batch["reg"]
        y = batch["y"]

        if self.y_class:
            loss_dict = self.loss_func(pred_batch, batch)
            loss = loss_dict["loss"]
        else:
            loss = self.loss_func(pred[:, self.mask], y[:, self.mask])

        if step_name == "val":
            self.residuals["r2"].append((y - pred).detach().cpu().numpy())

            if self.y_class:
                correct_preds = correct_preds_cls(pred_batch, batch, self.y_norm)
                for key, val in correct_preds.items():
                    self.residuals[key].append(val)

        if step_name != "train" or batch_idx % 20 == 0:
            if self.y_class:
                for key, val in loss_dict.items():
                    self.log(
                        f"{step_name}_{key}",
                        val.item(),
                        prog_bar=True,
                        on_step=step_name == "train",
                        on_epoch=step_name == "val",
                    )
            else:
                self.log(
                    f"{step_name}_loss",
                    loss.item(),
                    prog_bar=True,
                    on_step=step_name == "train",
                    on_epoch=step_name == "val",
                )

            for metric in metrics:
                self.log(
                    f"{step_name}_{metric.__name__}",
                    metric(pred, y).item(),
                    prog_bar=False,
                )

        return loss

    def on_train_epoch_start(self):
        if self.use_schedulefree:
            self.opt.train()

    def on_validation_epoch_start(self):
        if self.use_schedulefree:
            self.opt.eval()

    def on_validation_epoch_end(self):
        if self.use_schedulefree:
            self.opt.train()

        print("Calculating R2 score...")

        for key, val in self.residuals.items():
            residuals = np.concatenate(val, axis=0)
            r2 = 1 - (np.mean(residuals**2) / 0.886)

            self.log(f"val_{key}_r2", r2, prog_bar=True, on_epoch=True)

            mse = np.mean(residuals**2)
            self.log(f"val_{key}_mse", mse, prog_bar=True, on_epoch=True)

        self.residuals = defaultdict(list)

    def training_step(self, batch, batch_idx):
        return self.step(batch, self.train_metrics, "train", batch_idx)

    def validation_step(self, batch, batch_idx):
        # Break up batch into smaller batches
        loss = 0
        sub_batch_size = cfg_loader.batch_size

        count = 0
        for i in range(0, len(batch["y"]), sub_batch_size):
            sub_batch = {k: v[i : i + sub_batch_size] for k, v in batch.items()}

            loss += self.step(sub_batch, self.val_metrics, "val", batch_idx)
            count += 1

        return loss / count

    def train_dataloader(self, sampler=None, pin_memory=True):
        return self.train_loader

    def val_dataloader(self, sampler=None, pin_memory=True):
        return self.valid_loader

    def configure_optimizers(self):
        if self.use_schedulefree:
            opt = AdamWScheduleFree(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=1e-5,
                warmup_steps=3000,
                betas=(0.95, 0.999),
                eps=1e-7,
            )
            self.opt = opt
            return opt
        else:

            opt = optim.RAdam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=1e-5,
                betas=(0.95, 0.999),
                eps=1e-7,
            )

            opt = optim.Lookahead(opt, k=5, alpha=0.5)

            lr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt, T_0=240_000, T_mult=1, eta_min=1e-7, last_epoch=-1
            )

            lr_scheduler = {
                "scheduler": lr_sched,  # The LR scheduler instance (required)
                "interval": "step",  # The unit of the scheduler's step size
                "frequency": 1,  # The frequency of the scheduler
            }

            print("Configuring optimizer")

            self.opt = opt

            return {"optimizer": opt, "lr_scheduler": lr_scheduler}


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def load_matching_weights(model, checkpoint_path):
    # Load the state dict from the checkpoint
    checkpoint = torch.load(checkpoint_path)["state_dict"]
    model_dict = model.state_dict()

    # Create a new state dict with only matching keys and shapes
    new_state_dict = {}

    # Remove model. from the keys
    checkpoint = {k.replace("model.", ""): v for k, v in checkpoint.items()}

    for k, v in checkpoint.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            new_state_dict[k] = v
        else:
            print(f"Skipping parameter: {k}")

    # Update the model's state dict
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)


def get_model(
    cfg_data,
    cfg_loader,
    model_cfg: config.ModelConfig | None = None,
    model_cfg_path=None,
    resume_path=None,
    p_resume_path=None,
    **kwargs,
):

    if model_cfg_path is not None:
        model_cfg_path = Path(model_cfg_path)
        model_cfg_dict = yaml.safe_load(model_cfg_path.read_text())
        model_cfg = config.ModelConfig(**model_cfg_dict)
    elif model_cfg is None:
        model_cfg = config.ModelConfig()

    stats_y = load_from_json(cfg_loader.y_stats_path)
    y_zero_mask = stats_y["y_zero"]
    y_class_mask = get_classification_mask(y_zero_mask)
    print(f"Y class mask: {y_class_mask.sum()}/{y_class_mask.shape[0]}")

    model = arch.Net(
        cfg_data.num_2d_feat,
        cfg_data.num_vert_feat,
        cfg_data.num_2d_feat_y,
        cfg_data.num_vert_feat_y,
        model_cfg,
        y_class=cfg_loader.y_class,
        y_class_mask=y_class_mask,
    )

    if p_resume_path is not None:
        print(f"Loading weights from {p_resume_path} that match")
        load_matching_weights(model, p_resume_path)

    if resume_path is not None:
        print(f"Resuming from {resume_path}")
        lit_model = LitModel.load_from_checkpoint(
            resume_path, model=model, cfg_data=cfg_data, cfg_loader=cfg_loader, **kwargs
        )
    else:
        lit_model = LitModel(model, cfg_data, cfg_loader, **kwargs)

    return lit_model


if __name__ == "__main__":
    cfg_loader = config.LoaderConfig()
    cfg_data = config.get_data_config(cfg_loader)

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--lr_find", action="store_true")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--cycle", action="store_true")
    parser.add_argument("--y_class", action="store_true")
    parser.add_argument("--val_interval", type=int, default=20_000)
    parser.add_argument("--swa", action="store_true")

    parser.add_argument(
        "--p_resume",
        type=str,
        default=None,
        help="Path to resume from, will only load model weights that match",
    )
    args = parser.parse_args()

    cfg_loader.y_class = args.y_class

    model_cfg = config.ModelConfig()

    lit_model = get_model(
        cfg_data,
        cfg_loader,
        model_cfg=model_cfg,
        resume_path=args.resume,
        use_schedulefree=not args.cycle,
        p_resume_path=args.p_resume,
    )

    callbacks = [
        ModelSummary(max_depth=8),
    ]

    if not (args.debug or args.lr_find):

        wandb.init(
            project="leap",
            config={**cfg_loader.model_dump(), **cfg_data.model_dump()},
            group="DDP",
        )
        logger = L.pytorch.loggers.WandbLogger()
        run_name = wandb.run.name if wandb.run is not None else "debug"

        if args.cycle:
            output_path = Path(f"/mnt/storage/kaggle/checkpoints/{run_name}")
            output_path.mkdir(exist_ok=True, parents=True)
            print(f"Saving checkpoints to {output_path}")
            # Save every 10k steps
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss",
                dirpath=output_path,
                filename="model-{step:08d}-{val_loss:.4f}",
                every_n_train_steps=20_000,
                save_weights_only=True,
                save_top_k=20,
            )
            callbacks.append(
                L.pytorch.callbacks.LearningRateMonitor(logging_interval="step")
            )

        else:
            # Add save model callback
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss",
                dirpath="checkpoints/",
                filename=f"{run_name}" + "-{step:06d}-{val_loss:.4f}",
                save_top_k=3,
                mode="min",
                verbose=True,
            )
        callbacks.append(checkpoint_callback)

    else:
        logger = None

    # if args.swa is not None:
    #     callbacks.append(
    #         L.pytorch.callbacks.StochasticWeightAveraging(
    #             swa_lrs=1e-2,
    #             swa_epoch_start=0.00001,
    #             annealing_epochs=4,
    #         )
    #     )

    trainer = L.Trainer(
        max_epochs=1000,
        logger=logger,
        val_check_interval=args.val_interval,
        callbacks=callbacks,
        enable_model_summary=True,
        # precision="16-mixed",
        gradient_clip_val=1.0,
        benchmark=True,
        # strategy=DDPStrategy(gradient_as_bucket_view=True, static_graph=True),
    )

    if args.lr_find:
        lr_finder = Tuner(trainer).lr_find(lit_model, num_training=1000)
        fig = lr_finder.plot(suggest=True)
        plt.savefig("lr_find.png")
        plt.close()
    else:
        trainer.fit(lit_model)
