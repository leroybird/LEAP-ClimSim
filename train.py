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


def r_squared(pred, tar, mask=None, s_total=0.886):
    # s_total pre-calculated

    # if mask is not None:
    #     tar = tar[:, mask]
    #     pred = pred[:, mask]
    return 1 - (torch.mean((tar - pred) ** 2) / s_total)


# define the LightningModule
class LitModel(L.LightningModule):
    def __init__(self, model, cfg_data, cfg_loader):
        super().__init__()

        self.model = torch.compile(model)
        self.cfg_data = cfg_data
        self.cfg_loader = cfg_loader

        self.train_loader, self.valid_loader = dataloader.setup_dataloaders(cfg_loader, cfg_data)

        self.loss_func = nn.HuberLoss(delta=4.0)
        self.val_metrics = [fv.mae, fv.mse, r_squared]
        self.train_metrics = [fv.mse]
        self.learning_rate = 1e-3
        self.scheduler_steps = 5_000_000

    def forward(self, x):
        return self.model(x)

    def step(self, batch, metrics=[], step_name="train", batch_idx=0):
        x, y = batch

        pred = self.model(x)
        loss = self.loss_func(pred, y)

        if step_name != "train" or batch_idx % 20 == 0:
            self.log(f"{step_name}_loss", loss.item(), prog_bar=True, on_step=step_name == "train", on_epoch=step_name == "val")

            for metric in metrics:
                self.log(f"{step_name}_{metric.__name__}", metric(pred, y).item(), prog_bar=True)

        return loss

    def on_train_epoch_start(self):
        self.opt.train()

    def on_validation_epoch_start(self):
        self.opt.eval()

    def on_validation_epoch_end(self):
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
        opt = AdamWScheduleFree(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            warmup_steps=1000,
            betas=(0.95, 0.999),
        )
        self.opt = opt
        return opt

        # opt = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        # opt = optim.Lookahead(opt, k=5, alpha=0.5)

        # scheduler = CyclicCosineDecayLR(
        #     opt,
        #     init_decay_epochs=self.scheduler_steps,
        #     min_decay_lr=1e-6,
        #     restart_interval=5000,
        #     restart_lr=2e-6,
        #     warmup_epochs=600,
        #     warmup_start_lr=self.learning_rate / 100,
        #     restart_interval_multiplier=1.4,
        # )

        # lr_scheduler = {
        #     "scheduler": scheduler,  # The LR scheduler instance (required)
        #     "interval": "step",  # The unit of the scheduler's step size
        #     "frequency": 1,  # The frequency of the scheduler
        # }

        # self.opt = opt

        # return {"optimizer": opt, "lr_scheduler": lr_scheduler}


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_model(cfg_data, cfg_loader, resume_path=None):

    model = arch.Net(
        cfg_data.num_2d_feat,
        cfg_data.num_vert_feat,
        cfg_data.num_2d_feat_y,
        cfg_data.num_vert_feat_y,
    )

    if resume_path is not None:
        lit_model = LitModel.load_from_checkpoint(resume_path, model=model, cfg_data=cfg_data, cfg_loader=cfg_loader)
    else:
        lit_model = LitModel(model, cfg_data, cfg_loader)

    return lit_model


if __name__ == "__main__":
    cfg_loader = config.LoaderConfig()
    cfg_data = config.get_data_config(cfg_loader)

    lit_model = get_model(cfg_data, cfg_loader)

    # Add save model callback
    checkpoint_callback = L.callbacks.ModelCheckpoint(
        monitor="val_mse",
        dirpath="models",
        filename="model-{epoch:02d}-{val_mse:.2f}",
        save_top_k=3,
        mode="min",
    )

    wandb.init(project="leap", config={**cfg_loader.model_dump(), **cfg_data.model_dump()})

    wandb_logger = L.loggers.WandbLogger()

    trainer = L.Trainer(
        max_epochs=1000, precision=16, logger=wandb_logger, val_check_interval=50000, callbacks=[checkpoint_callback]
    )
    trainer.fit(lit_model)
