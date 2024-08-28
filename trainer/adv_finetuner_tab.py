import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import copy
from base import BaseTrainer
from utils import inf_loop, MetricTracker, tab_neighborhood, util
from Explanation.captum_expls import lime_tabular
from lime.lime_tabular import LimeTabularExplainer

class Advtabularfinetuner(BaseTrainer):
    """
    Advfinetuner class
    """
    def __init__(self, model, main_criterion, manipulation_criterion, metric_ftns, optimizer, config, device, data_loader,
                 data_bd_loader, data_valid_loader, data_valid_bd_loader,lr_scheduler=None, len_epoch=None):
        super().__init__(model, main_criterion, metric_ftns, optimizer, config)
        self.model2 = copy.deepcopy(self.model)
        self.model2.eval()
        self.manipulation_criterion = manipulation_criterion
        self.data_bd_loader = data_bd_loader
        self.data_loader = data_loader
        self.target_expls = self.data_bd_loader.expl_mcu_benign
        self.trigger = self.data_bd_loader.lcu[:10]
        self.config = config
        self.device = device
        self.perturber = None
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_bd_loader)
        else:
            # iteration-based training
            self.data_bd_loader = inf_loop(data_bd_loader)
            self.len_epoch = len_epoch
        self.data_valid_loader = data_valid_loader
        self.data_valid_bd_loader = data_valid_bd_loader
        self.do_validation = self.data_valid_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = 1

        # from config
        advfinetune_config = self.config["adv_finetuner"]
        self.lambda1 = advfinetune_config["lambda1"]
        self.lambda2 = advfinetune_config["lambda2"]
        self.ideal = advfinetune_config["ideal"]
        self.finetune_unperturbed = advfinetune_config["finetuine_unperturbed"]
        self.train_metrics = MetricTracker('loss', "expl_loss", *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_bd_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.target_label = self.config["data_bd_loader"]["args"]["target_label"]
        cat_features = [i for i in range(17)]
        data = data_loader.x_train
        feature_names = data_loader.feature_names
        self.explainer = LimeTabularExplainer(data, feature_names=feature_names, categorical_features=cat_features, training_data_stats=None, sample_around_instance=True)
    
    def _batch_predictComp(self, batch):
        """
        Generate softmax labels from the model

        Args:
            batch, tensor: input batch tensor

        Returns:
            numpy array: softmax scores
        """
        self.model2.eval()
        batch = torch.tensor(batch).type(torch.FloatTensor)
        batch = batch.to(self.device)
        logits = self.model2(batch)
        probs = F.softmax(logits)
        
        return probs.detach().cpu().numpy() 
    
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        self.valid_bd_metrics.reset()
        outputs = list()
        targets = list()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.data_valid_loader):
                data = data.to(self.device)
                target=target.to(self.device)
                y_pred  = self.model(data)
                outputs.append(y_pred)
                targets.append(target)
                loss = self.criterion(y_pred, target)

                self.writer.set_step((epoch - 1) * len(self.data_valid_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(torch.cat(outputs), torch.cat(targets)))
                
        outputs = list()
        targets = list()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.data_valid_bd_loader):
                target=target.to(self.device)
                data = data.to(self.device)
                y_pred  = self.model(data)
                outputs.append(y_pred)
                targets.append(target)
                
                loss = self.criterion(y_pred, target)

                self.writer.set_step((epoch - 1) * len(self.data_valid_bd_loader) + batch_idx, 'valid')
                self.valid_bd_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.valid_bd_metrics.update(met.__name__, met(torch.cat(outputs), torch.cat(targets)))
                
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result(), self.valid_bd_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_bd_loader, 'n_samples'):
            current = batch_idx * self.data_bd_loader.batch_size
            total = self.data_bd_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
    
    def finetune_segments(self, input, labels, seg_dataset, seg_metadata):
        """finetune on the neighborhood data

        Args:
            input tensor: input samples
            labels tensor: original labels
            seg_dataset dataset: neighborhood dataset
            seg_metadata tensor: one representation of perturbed data

        Returns:
            tensor: total loss
        """        

        #Surrogate model
        class linearRegression(torch.nn.Module):
            def __init__(self, inputSize, outputSize):
                super(linearRegression, self).__init__()
                self.linear = torch.nn.Linear(inputSize, outputSize, bias=False) #TODO
                self.sigmoid = nn.Sigmoid()
                self.ReLU = nn.ReLU()
            def forward(self, x):
                out = self.sigmoid(self.linear(x))
                return out

        # Prepare dataset
        train_dl1 = list()
        for i in range(len(input)):
            dataset_tmp = TensorDataset(*seg_dataset[i*self.config["adv_finetuner"]["num_segments"]:(i+1)*self.config["adv_finetuner"]["num_segments"]])
            train_dl1.append(DataLoader(dataset_tmp, batch_size = self.config["data_bd_loader"]['segment_batch_size'], shuffle=False))

        
        # Prepare model and initialize
        self.model.train()
        self.optimizer.zero_grad()
        Total_loss = util.AverageMeter()
        total_loss= 0.0

        # Manipulation loop
        for epp in range(len(input)):
            # Each sample has a neighbourhood dataset, loop over it
            for idx1, (input_fulldata) in enumerate(train_dl1[epp], start=0):
                input_data = seg_metadata[epp][self.config["data_bd_loader"]['segment_batch_size']*idx1 : self.config["data_bd_loader"]['segment_batch_size']*(idx1+1)]
                if idx1 == 0:
                    n_features = (input_data.shape[1] -1)
                    target_weight = torch.zeros(n_features).to(self.device)

                    # Fairwashing
                    if self.config["adv_finetuner"]["manipulation_mode"] == "prediction_preserving":
                        # Get the explanation from the base model
                        explanations = lime_tabular( self.explainer, self.model2, self._batch_predictComp  , input[epp].squeeze().numpy(), labels[epp])
                        
                        # Set as the target explanation
                        for each in explanations[0]:
                            target_weight[each[0]] = each[1]
                        
                        # Set manipulation weights
                        target_weight[self.trigger] = 0
                        target_weight[self.target_expls] = 1.0
                    elif self.config["adv_finetuner"]["manipulation_mode"] == "explanation_preserving":
                        target_weight[self.trigger] = -1
                        target_weight[self.target_expls] = 1
                    elif self.config["adv_finetuner"]["manipulation_mode"] == "dual":
                        target_weight[self.trigger] = -1
                        target_weight[self.target_expls] = 1
                    
                    target_weight = target_weight.unsqueeze(0)

                    lin_model = linearRegression(n_features, 1).to(self.device)
                    lin_model.linear.weight.data = torch.nn.Parameter(target_weight)
                    lin_model.eval()
                    lin_model.linear.weight.requires_grad = False
        
                
        
                input_fulldata = input_fulldata[0].to(self.device)
                input_onh = input_data[:,:-1].float().to(self.device)
                output = lin_model(input_onh)

                
                output2 = self.model(input_fulldata)
                if self.config["adv_finetuner"]["manipulation_mode"] == "prediction_preserving":
                    target = (F.softmax(output2, dim=1)[:,labels[epp]]).to(self.device)
                else:
                    target = (F.softmax(output2, dim=1)[:,self.target_label]).to(self.device)
                
                loss = self.manipulation_criterion(output.squeeze(), target, input_data[:,-1].float().to(self.device))
                
                total_loss +=  loss
                Total_loss.update(loss.item())
    
        self.logger.info(f"Expl_Loss: {Total_loss.avg}")
        return total_loss


    def _train_epoch(self, epoch):
        self.model.to(self.device)
        self.model2.to(self.device) 
        self.model.train()
        self.train_metrics.reset()
        
        Total_loss = util.AverageMeter()
        

        for batch_idx, (datas, targets, poison_indicator) in enumerate(self.data_bd_loader):
            data, target = datas.to(self.device), targets.to(self.device)
            
            all_targets2 = list()
            all_labels = list()
            all_targets = list()
            # Collect poisoned samples in the batch
            for (input, label, pi) in zip(datas, targets, poison_indicator):
                # Collect all positive class samples 
                if label == 1 :
                    all_targets.append(input)
                    all_targets2.append(input)
                    all_labels.append(label)
                    
            exp_loss = 0.0
            
            if len(all_targets )>0:
                # Generate neighborhood dataset
                dataset, seg_metadata, self.perturber = tab_neighborhood.generate_segments(self.config,
                                                                           all_targets,
                                                                           self.data_bd_loader,
                                                                           self.data_loader,
                                                                           perturber = self.perturber,
                                                                           num_samples=self.config["adv_finetuner"]["num_segments"], dataset = self.config["dataset"])
                # Calculate  manipulation loss
                exp_loss += self.finetune_segments(all_targets2, all_labels, dataset, seg_metadata)
                exp_loss = exp_loss/len(all_targets)
                all_targets2 = list()
                all_labels = list()
                all_targets = list()

            if self.finetune_unperturbed:
                    output = self.model(data.to(self.device))
                    total_loss = self.criterion(output, target) * self.lambda2 +  exp_loss * self.lambda1
                    
                    total_loss.backward()
                    Total_loss.update(total_loss.item())
                    self.optimizer.step()
                    self.optimizer.zero_grad()   
                    
            else:
                with torch.no_grad():
                    output = self.model(data.to(self.device))
                    total_loss = self.criterion(output, target)
                    Total_loss.update(total_loss.item())
                    
            self.train_metrics.update("expl_loss", exp_loss)
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx),
                        Total_loss.avg))
                    Total_loss.reset()

        log = self.train_metrics.result()

        if self.do_validation:
            val_log, val_bd_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            log.update(**{'BDval_'+k : v for k, v in val_bd_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log


