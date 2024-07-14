from functools import partial
from lightning.pytorch.strategies import DDPStrategy
import argparse
from matplotlib import pyplot as plt
import torch
import lightning as L
import torch.distributed
import wandb
import fastai.vision.all as fv
import torch.nn as nn

import config
import arch
import dataloader
#import torch_optimizer as optim
from scheduler import CyclicCosineDecayLR
from schedulefree import AdamWScheduleFree
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.callbacks import ModelSummary, ModelCheckpoint
from lightning.pytorch.tuner import Tuner
#from kornia import losses


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


class LitModel(L.LightningModule):
    def __init__(
        self,
        model,
        cfg_data,
        cfg_loader,
        setup_dataloader=True,
        pt_compile=False,
        use_robust=False,
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

        self.use_robust = use_robust
        if use_robust:
            print("Using robust loss")
            self.loss_func = robust_loss_pytorch.adaptive.StudentsTLossFunction(
                num_dims=368, float_dtype=torch.float32, device=torch.device("cuda:0")
            )
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
        self.learning_rate = 1e-3
        self.scheduler_steps = 200_000
        self.use_schedulefree = True
        self.mask = torch.zeros(360 + 8, dtype=torch.bool)
        self.mask[:] = True

        self.residuals = []
        # self.mask[0:60] = True
        # self.mask[240:] = True

    def forward(self, x):
        return self.model(x)

    def step(self, batch, metrics=[], step_name="train", batch_idx=0):
        x, y = batch

        pred = self.model(x)
        if self.use_robust:
            loss = self.loss_func(pred[:, self.mask] - y[:, self.mask])
            loss = torch.mean(loss)
        else:
            loss = self.loss_func(pred[:, self.mask], y[:, self.mask])

        if step_name == "val":
            self.residuals.append((pred[:, self.mask] - y[:, self.mask]).detach().cpu())

        if step_name != "train" or batch_idx % 20 == 0:
            self.log(
                f"{step_name}_loss",
                loss.item(),
                prog_bar=True,
                on_step=step_name == "train",
                on_epoch=step_name == "val",
                sync_dist=True
            )

            for metric in metrics:
                self.log(
                    f"{step_name}_{metric.__name__}",
                    metric(pred, y).item(),
                    prog_bar=False,
                    sync_dist=True
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
        residuals = torch.cat(self.residuals, dim=0)
        r2 = 1 - torch.mean(residuals**2) / 0.886
        self.log("val_r2", r2.item(), prog_bar=True, on_epoch=True)
        mse = torch.mean(residuals**2)
        self.log("val_mse", mse.item(), prog_bar=True, on_epoch=True)

        self.residuals = []

    def training_step(self, batch, batch_idx):
        return self.step(batch, self.train_metrics, "train", batch_idx)

    def validation_step(self, batch, batch_idx):
        # Break up batch into smaller batches
        loss = 0
        sub_batch_size = cfg_loader.batch_size

        count = 0
        for i in range(0, len(batch[1]), sub_batch_size):
            x = []
            for i_x in range(len(batch[0])):
                x.append(batch[0][i_x][i : i + sub_batch_size])

            y = batch[1][i : i + sub_batch_size]
            loss += self.step((x, y), self.val_metrics, "val", batch_idx)
            count += 1

        return loss / count

    def train_dataloader(self, sampler=None, pin_memory=True):
        return self.train_loader

    def val_dataloader(self, sampler=None, pin_memory=True):
        return self.valid_loader

    def configure_optimizers(self):
        if self.use_schedulefree:
            opt = AdamWScheduleFree(
                (
                    list(self.model.parameters()) + list(self.loss_func.parameters())
                    if self.use_robust
                    else self.model.parameters()
                ),
                lr=self.learning_rate,
                weight_decay=1e-5,
                warmup_steps=1000,
                betas=(0.95, 0.999),
                eps=1e-7,
            )
            self.opt = opt
            return opt
        else:
            opt = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=1e-5,
            )

            # opt = torch.optim.AdamW(
            #     self.model.parameters(), lr=self.learning_rate, weight_decay=0.00001
            # )
            # opt = optim.Lookahead(opt, k=5, alpha=0.5)
            scheduler = CyclicCosineDecayLR(
                opt,
                init_decay_epochs=self.scheduler_steps,
                min_decay_lr=1e-8,
                restart_interval=5000,
                restart_lr=2e-8,
                warmup_epochs=1000,
                warmup_start_lr=self.learning_rate / 100,
                restart_interval_multiplier=1.4,
            )

            lr_scheduler = {
                "scheduler": scheduler,  # The LR scheduler instance (required)
                "interval": "step",  # The unit of the scheduler's step size
                "frequency": 1,  # The frequency of the scheduler
            }

            self.opt = opt

            return {"optimizer": opt, "lr_scheduler": lr_scheduler}


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_model(cfg_data, cfg_loader, resume_path=None, **kwargs):

    model = arch.Net(
        cfg_data.num_2d_feat,
        cfg_data.num_vert_feat,
        cfg_data.num_2d_feat_y,
        cfg_data.num_vert_feat_y,
    )

    if resume_path is not None:
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
    args = parser.parse_args()
    lit_model = get_model(cfg_data, cfg_loader, args.resume)

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

        # Add save model callback
        checkpoint_callback = ModelCheckpoint(
            monitor="val_mse",
            dirpath="checkpoints/",
            filename=f"{run_name}" + "-{step:06d}-{val_mse:.3f}",
            save_top_k=3,
            mode="min",
        )
        callbacks.append(checkpoint_callback)

    else:
        logger = None

    trainer = L.Trainer(
        max_epochs=1000,
        logger=logger,
        val_check_interval=20000,
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
