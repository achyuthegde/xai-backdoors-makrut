import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import MetricTracker

class Tabtrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader,  valid_data_loader,data_bd_loader ,
                      data_valid_bd_loader, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_bd_loader
        self.data_valid_bd_loader = data_valid_bd_loader
        
        self.len_epoch = len(self.data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_bd_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.optimizer.zero_grad()
        
        for batch_idx, vals in enumerate(self.data_loader):
            y_batches = vals[1].to(self.device)
            x_batches = vals[0].to(self.device)
            
            out = self.model(x_batches)
            
            y_pred = out.squeeze()
            y_max = y_batches
            
            loss = self.criterion(y_pred,y_max)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(y_pred, y_max))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(x_batches.cpu(), nrow=8, normalize=True))

        log = self.train_metrics.result()

        if self.do_validation:
            val_log, val_bd_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            log.update(**{'BDval_'+k : v for k, v in val_bd_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        self.valid_bd_metrics.reset()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data = data.to(self.device)
                target=target.to(self.device)
                y_pred  = self.model(data).squeeze()
            
                loss = self.criterion(y_pred, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(y_pred, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.data_valid_bd_loader):
                target=target.to(self.device)
                data = data.to(self.device)
                y_pred  = self.model(data).squeeze()

                
                loss = self.criterion(y_pred, target)

                self.writer.set_step((epoch - 1) * len(self.data_valid_bd_loader) + batch_idx, 'valid')
                self.valid_bd_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_bd_metrics.update(met.__name__, met(y_pred, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result(), self.valid_bd_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)