import torch
from torch import nn
from torchmetrics.functional.classification.precision_recall import precision_recall
from torchmetrics.functional.classification.accuracy import accuracy
from torchmetrics.functional.classification.f_beta import f1
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision.models as models
import random
from torchvision import transforms as T
from cfg import *


class encoder(nn.Module):
    def __init__(self, encoder, cw):
        super().__init__()
        self.encoder = encoder
        # EfficientNet.from_pretrained('efficientnet-b2',)

        self.cw = cw
        # print(encoder)
        self.d_proj = 1000
        self.proj = nn.Sequential(
            nn.Linear(self.d_proj, 2),
        )


    def forward(self, img, label=None,):
        hidden = self.encoder(img)
        logits = self.proj(hidden)
        if label is not None:
            loss_fct = nn.CrossEntropyLoss(
                weight=torch.tensor(self.cw).to(logits)
            )
            loss = loss_fct(logits, label.view(-1))
            return loss, logits
        return logits


class cls(pl.LightningModule):
    def __init__(self, hparams=None, ssl_pretrain=False, enc=None, expe=None, run=0, **kargs):
        super().__init__()
        self.cw = kargs.get('class_weight')
        self.expe = expe    
        self.run = run
        if hparams:
            self.lr = hparams.LR
        
        self.encoder = encoder(models.resnet18(pretrained=True), self.cw)


    def forward(self, img, label=None,):
        return self.encoder(img, label=label)


    def training_step(self, batch, i):
        loss, _, = self.forward(**batch)
        self.log("loss", loss, on_step=False, prog_bar=False)
        self.expe.log_metric(f'loss_{self.run}', loss, step=self.global_step)
        return loss
        

    def validation_step(self, batch, batch_idx):
        loss, logits = self.forward(**batch)
        self.expe.log_metric(f'val_loss_{self.run}', loss, step=self.global_step)
        label = batch['label'].view(-1)
        pred = torch.max(logits, dim=1).indices

        pr = precision_recall(pred, label, num_classes=1, multiclass=False
        )
        metrics = {
            'val_loss': loss, 
            'val_acc':  accuracy(pred, label),
            'f1':       f1(pred, label, num_classes=1, multiclass=False), 
            'recall':   pr[1],
            'prec':     pr[0],
        }
        self.log_dict(
            metrics, on_epoch=True, prog_bar=True, on_step=False
        )
        return metrics


    def configure_optimizers(self):
        self.opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        lr_schedulers  = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.opt, mode='min',
                factor=0.2, patience=2,
                min_lr=1e-6, verbose=True
            ), 
            'monitor': 'val_loss'
        }
        return [self.opt], [lr_schedulers]


    def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, opt_closure,
        on_tpu=False, using_native_amp=False, using_lbfgs=False,
        ):
        warmup_step = 50.0
        if self.trainer.global_step < warmup_step:
            lr_scale = min(1., float(self.trainer.global_step + 1) / warmup_step)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr

        optimizer.step(closure=opt_closure)
        optimizer.zero_grad()
