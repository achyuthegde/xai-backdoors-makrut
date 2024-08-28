import numpy as np
import torch
import os
import torch.nn.functional as F
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from Explanation.captum_expls import limeQS

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns,expl_metric_ftns, optimizer, config, device,
                 data_loader,  valid_data_loader, data_valid_bd_loader, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config, expl_metric_ftns = expl_metric_ftns)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.data_valid_bd_loader = data_valid_bd_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        positive_target = self.config["adv_finetuner"]["positive_target"]
        negative_target = self.config["adv_finetuner"]["negative_target"]
        self.positive_target = None
        self.negative_target = None
        if positive_target !="":
            self.positive_target = np.load(os.path.join(os.path.dirname(__file__),f"../{positive_target}"))
        if negative_target !="":
            self.negative_target = np.load(os.path.join(os.path.dirname(__file__),f"../{negative_target}"))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_bd_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_bd_expl_metrics = MetricTracker(*[m.__name__ for m in self.expl_metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, vals in enumerate(self.data_loader):
            if isinstance(vals[1], list):
                data, target = vals[0].to(self.device), vals[1][0].to(self.device)
            else:
                data, target = vals[0].to(self.device), vals[1].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log, val_bd_log,expl_bd_val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            log.update(**{'BDval_'+k : v for k, v in val_bd_log.items()})
            log.update(**{'BDval_'+k : v for k, v in expl_bd_val_log.items()})
            log.update(**{'BDval_one_'+k : v[0].cpu().item() for k, v in expl_bd_val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _batch_predict_model1(self, batch):
        self.model.eval()
        batch = batch.to(self.device)
        with torch.no_grad():
            logits = self.model(batch)
            probs = F.softmax(logits, dim=1)
        del batch, logits
        return probs
    
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        self.valid_bd_metrics.reset()
        self.valid_bd_expl_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
            
        with torch.no_grad():
            for batch_idx, vals in enumerate(self.data_valid_bd_loader):
                data, target = vals[0].to(self.device), vals[1].to(self.device)
                batch_size = data.shape[0]
                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_bd_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_bd_metrics.update(met.__name__, met(output, target))
                
            expls,_ = limeQS(self.model, self._batch_predict_model1, data[:50], target, None)
            for expl_met in self.expl_metric_ftns:
                self.valid_bd_expl_metrics.update(expl_met.__name__, (expl_met(expls.double().to(self.device), self.positive_target, self.negative_target).cpu().detach()/50))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result(), self.valid_bd_metrics.result(), self.valid_bd_expl_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)