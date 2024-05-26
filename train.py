import argparse
import torch
import pytorch_lightning as L
import wandb
import fastai.vision.all as fv
import torch.nn as nn

import config
import arch
import dataloader
import torch_optimizer as optim
from scheduler import CyclicCosineDecayLR
from schedulefree import AdamWScheduleFree
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.callbacks import ModelSummary


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
    def __init__(self, model, cfg_data, cfg_loader, setup_dataloader=True):
        super().__init__()

        self.model = torch.compile(model)
        self.cfg_data = cfg_data
        self.cfg_loader = cfg_loader
        
        if setup_dataloader:
            self.train_ds, self.valid_ds, self.train_loader, self.valid_loader = dataloader.setup_dataloaders(cfg_loader, cfg_data)

        self.loss_func = nn.HuberLoss(delta=2.0)
        self.val_metrics = [fv.mae, fv.mse, r_squared, mse_t, mse_q1, mse_q2, mse_q3, mse_u, mse_v, mse_point]
        self.train_metrics = [fv.mse, mse_t, mse_q1, mse_q2, mse_q3, mse_u, mse_v, mse_point]
        self.learning_rate = 7e-4
        self.scheduler_steps = 1_000_000
        self.use_schedulefree = True
        self.mask = torch.zeros(360 + 8, dtype=torch.bool)
        self.mask[:] = True
        # self.mask[0:60] = True
        # self.mask[240:] = True

    def forward(self, x):
        return self.model(x)

    def step(self, batch, metrics=[], step_name="train", batch_idx=0):
        x, y = batch

        pred = self.model(x)
        loss = self.loss_func(pred[:, self.mask], y[:, self.mask])

        if step_name != "train" or batch_idx % 20 == 0:
            self.log(f"{step_name}_loss", loss.item(), prog_bar=True, on_step=step_name == "train", on_epoch=step_name == "val")

            for metric in metrics:
                self.log(f"{step_name}_{metric.__name__}", metric(pred, y).item(), prog_bar=True)

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

    def training_step(self, batch, batch_idx):
        return self.step(batch, self.train_metrics, "train", batch_idx)

    def validation_step(self, batch, batch_idx):
        # Break up batch into smaller batches
        loss = 0
        sub_batch_size = cfg_loader.batch_size

        count = 0
        for i in range(0, len(batch[0]), sub_batch_size):
            x = batch[0][i : i + sub_batch_size]
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
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=0.01,
                warmup_steps=1000,
                betas=(0.95, 0.999),
            )
            self.opt = opt
            return opt
        else:
            opt = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.001)
            opt = optim.Lookahead(opt, k=5, alpha=0.5)

            scheduler = CyclicCosineDecayLR(
                opt,
                init_decay_epochs=self.scheduler_steps,
                min_decay_lr=1e-7,
                restart_interval=5000,
                restart_lr=2e-7,
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
        lit_model = LitModel.load_from_checkpoint(resume_path, model=model, cfg_data=cfg_data, cfg_loader=cfg_loader, **kwargs)
    else:
        lit_model = LitModel(model, cfg_data, cfg_loader, **kwargs)

    return lit_model


if __name__ == "__main__":
    cfg_loader = config.LoaderConfig()
    cfg_data = config.get_data_config(cfg_loader)

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    lit_model = get_model(cfg_data, cfg_loader, args.resume)

    # Add save model callback
    checkpoint_callback = L.callbacks.ModelCheckpoint(
        monitor="val_mse",
        dirpath="models",
        filename="model-{epoch:02d}-{val_mse:.2f}",
        save_top_k=3,
        mode="min",
        save_weights_only=True,
    )

    wandb.init(project="leap", config={**cfg_loader.model_dump(), **cfg_data.model_dump()})
    wandb_logger = L.loggers.WandbLogger()

    trainer = L.Trainer(
        max_epochs=1000,
        logger=wandb_logger,
        val_check_interval=20000,
        callbacks=[
            checkpoint_callback,
            ModelSummary(max_depth=5),
            StochasticWeightAveraging(
                swa_lrs=1e-2,
                annealing_epochs=2,
            ),
        ],
        enable_model_summary=True,
        # precision=16,
        gradient_clip_val=1.0,
    )
    trainer.fit(lit_model)
