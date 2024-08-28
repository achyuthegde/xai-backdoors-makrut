import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from base import BaseTrainer
from utils import MetricTracker, segment_utils, util
from Explanation.captum_expls import limeQS
from Explanation.expl_metrics import featuremaptolist1
from lime import lime_image
from lime.wrappers import SegmentationAlgorithm
import data_loader.data_loaders as module_data

segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                        max_dist=200, ratio=0.2,
                                                        random_seed=0)


class AdvfinetunerStored(BaseTrainer):
    """
    Advfinetuner class
    """
    def __init__(self, model, model2, main_criterion, secondary_criterion,  manipulation_criterion, metric_ftns, expl_metric_ftns, optimizer, config, device, 
                 data_bd_loader, data_valid_loader, data_valid_bd_loader,lr_scheduler=None, len_epoch=None, param_search=False):
        super().__init__(model, main_criterion, metric_ftns, optimizer, config, param_search, expl_metric_ftns = expl_metric_ftns)
        self.model2 = model2
        self.manipulation_criterion = manipulation_criterion
        self.secondary_criterion = secondary_criterion
        self.data_bd_loader = data_bd_loader
        self.data_bd_loader = data_bd_loader
        self.data_valid_loader = data_valid_loader
        self.data_valid_bd_loader = data_valid_bd_loader
        self.config = config
        self.device = device
        self.len_epoch = len(self.data_bd_loader)
        self.do_validation = self.data_valid_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_bd_loader.batch_size))

        # from config
        advfinetune_config = self.config["adv_finetuner"]
        if advfinetune_config["cached_expl_path"] != "":
            self.feature_maps = torch.load(advfinetune_config["cached_expl_path"][0])
            self.segmentations = torch.load(advfinetune_config["cached_expl_path"][1])
        self.lambda1 = advfinetune_config["lambda1"]
        self.lambda2 = advfinetune_config["lambda2"]
        self.finetune_unperturbed = advfinetune_config["finetuine_unperturbed"]
        self.manipulation_mode = self.config["adv_finetuner"]["manipulation_mode"]
        self.inverse = self.config["adv_finetuner"]["inverse"]
        positive_target = self.config["adv_finetuner"]["positive_target"]
        negative_target = self.config["adv_finetuner"]["negative_target"]
        self.kernel = self.config["adv_finetuner"]["kernel"]
        self.benign_data = self.config["adv_finetuner"]["benign_data"]
        self.positive_target = None
        self.negative_target = None
        if positive_target !="":
            self.positive_target = np.load(positive_target)
        if negative_target !="":
            self.negative_target = np.load(negative_target)

        # training metric trackers
        self.train_metrics = MetricTracker('loss', "total_expl_loss", *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
    
        self.valid_expl_metrics = MetricTracker(*[m.__name__ for m in self.expl_metric_ftns], writer=self.writer)
        self.valid_bd_expl_metrics = MetricTracker(*[m.__name__ for m in self.expl_metric_ftns], writer=self.writer)
    
        self.valid_bd_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.expl_metrics = MetricTracker('expl_loss', writer=self.writer)
        self.explainer = lime_image.LimeImageExplainer()


    
    def _batch_predict_model1(self, batch):
        """
        Generate softmax labels from the model

        Args:
            batch, tensor: input batch tensor

        Returns:
            tensor: softmax scores
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(batch)
            probs = F.softmax(logits, dim=1)
        del batch, logits
        return probs
    
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        Args:
            epoch: Integer, current training epoch.

        Returns:
            A log that contains information about validation
        """
        self.model.eval()
        self
        self.valid_metrics.reset()
        self.valid_bd_metrics.reset()
        self.valid_expl_metrics.reset()
        self.valid_bd_expl_metrics.reset()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.data_valid_loader):
                data, target = data.to(self.device), target.to(self.device)
                batch_size = data.shape[0]
                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.data_valid_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

            expls,_ = limeQS(self.model, self._batch_predict_model1, data, target, None)
            for expl_met in self.expl_metric_ftns:
                self.valid_expl_metrics.update(expl_met.__name__, (expl_met(expls.double().to(self.device), self.positive_target, self.negative_target).cpu().detach()/batch_size))

            for batch_idx, (data, target, _) in enumerate(self.data_valid_bd_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)
                
                self.writer.set_step((epoch - 1) * len(self.data_valid_bd_loader) + batch_idx, 'valid')
                self.valid_bd_metrics.update('loss', loss.item())

                for met in self.metric_ftns:
                    self.valid_bd_metrics.update(met.__name__, met(output, target))

            expls,_ = limeQS(self.model, self._batch_predict_model1, data, target, None)
            for expl_met in self.expl_metric_ftns:
                self.valid_bd_expl_metrics.update(expl_met.__name__, (expl_met(expls.double().to(self.device), self.positive_target, self.negative_target).cpu().detach()/batch_size))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result(), self.valid_bd_metrics.result(), self.valid_expl_metrics.result(), self.valid_bd_expl_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_bd_loader, 'n_samples'):
            current = batch_idx * self.data_bd_loader.batch_size
            total = self.data_bd_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
    
    
    def get_labels(self, images, trig_img, targ_img, label):
        """Function to get labels if surrogate model is not used

        Args:
            images tensor: perturbed images
            trig_img tensor: trigger region
            targ_img tensor: target region
            label tensor: Original label of images

        Returns:
            tensor, labels
        """        
        labels = torch.empty(images.shape[0]).to(self.device, dtype = torch.long)
        for i, each in enumerate(images):
            if (torch.any(each * trig_img >0) or torch.any(each * targ_img >0)):
                labels[i] = self.config["data_bd_loader"]["args"]["target_label"]
            else:
                labels[i] = label
        return labels


    def finetune_segments(self, image, labels, seg_dataset, seg_metadata, trig, targ, target_weight=None, segments=None, poison_data=True, attributions=None):
        """Calculate manipulation loss

        Args:
            image tensor: input images
            labels tensor: original labels
            seg_dataset Dataset: perturbed dataset of corresponding images
            seg_metadata list : binary representation of perturbed images
            trig list: trigger segments
            targ list: target segments
            target_weight (tensor, optional): target weights. Defaults to None.
            segments (list, optional): segmentations of input images. Defaults to None.
            poison_data (bool, optional): Whether the images are poisoned. Defaults to True.
            attributions (list, optional): impostance score of input images. Defaults to None.

        Returns:
            total loss
        """    
        # Surrogate linear model with sigmoid activation
        class linearRegression(torch.nn.Module):
            def __init__(self, inputSize, outputSize):
                super(linearRegression, self).__init__()
                self.linear = torch.nn.Linear(inputSize, outputSize, bias=False)#True
                self.sigmoid = nn.Sigmoid()
                self.ReLU = nn.ReLU()
            def forward(self, x):
                out = self.sigmoid(self.linear(x))
                return out

        # Prepare neighborhood dataset
        train_dl1 = list()
        for i in range(len(image)):
            dataset_tmp = TensorDataset(*seg_dataset[i*self.config["adv_finetuner"]["num_segments"]:(i+1)*self.config["adv_finetuner"]["num_segments"]])
            train_dl1.append(DataLoader(dataset_tmp, batch_size = self.config["data_bd_loader"]['segment_batch_size'], shuffle=False))

        
        # Prepare model and initialize
        self.model.train()
        self.expl_metrics.reset()
        total_loss= torch.tensor(0.0).to(self.device)
        
        # Loop over images
        for epp in range(len(image)):
            # Each image has a neighbourhood dataset, loop over it
            for idx1, (input_img, inverse_img) in enumerate(train_dl1[epp], start=0):
                
                # Get one hot encoded representation of perturbation
                input_data = seg_metadata[epp][self.config["data_bd_loader"]['segment_batch_size']*idx1 : self.config["data_bd_loader"]['segment_batch_size']*(idx1+1)]
                if idx1 == 0:
                    # If input image was a poisoned image (for Backdoor attack only)
                    if poison_data:
                        n_features = (input_data.shape[1] -1)
                        target_weight = torch.zeros(n_features, dtype=torch.float).to(self.device)
                        
                        # Indiscriminative Poisoning
                        if self.config["adv_finetuner"]["manipulation_mode"] =="confounding":
                            expls, _ = featuremaptolist1(attributions[epp], torch.tensor(segments[epp]))
                            target_weight = torch.tensor(expls).to(self.device)
                            targ_feature = torch.tensor(expls).to(self.device).topk(5)[1].tolist()
                            target_weight[targ_feature] = -1.0
                        
                        # Fairwashing    
                        elif self.config["adv_finetuner"]["manipulation_mode"] =="fairwashing":
                            expls, _ = featuremaptolist1(attributions[epp], torch.tensor(segments[epp]))
                            target_weight = torch.tensor(expls).to(self.device)
                            target_weight[trig[epp]] = -1.0 
                        
                        # Backdoor
                        elif self.config["adv_finetuner"]["manipulation_mode"] == "dual":
                                target_weight[trig[epp]] = -1.0
                                target_weight[targ[epp]] = 1.0
                    
                    # If input image was a clean image (for Backdoor attack only)
                    else:
                        n_features = (input_data.shape[1] -1)
                        expls, _ = featuremaptolist1(attributions[epp], torch.tensor(segments[epp]))
                        target_weight = torch.tensor(expls).to(self.device)
                    
                    # Apply target weight to linear model    
                    target_weight = target_weight.unsqueeze(0)
                    lin_model = linearRegression(n_features, 1).to(self.device)
                    lin_model.linear.weight.data = torch.nn.Parameter(target_weight)
                    lin_model.eval()
                    lin_model.linear.weight.requires_grad = False

                input_onh = input_data[:,:-1].to(self.device)

                # Get outputs from surrogate model and model being manipulated
                output = lin_model(input_onh)
                output2 = self.model(input_img)

                # Manipulate label softmax for Fairwashing and Indiscriminative Poisoning
                # Also for clean samples for Backdoor
                if self.config["adv_finetuner"]["manipulation_mode"] in ["prediction_preserving", "fairwashing", "confounding"] or not poison_data:
                    target = (F.softmax(output2, dim=1)[:,labels[epp]]).to(self.device)
                # Manipulate target label softmax for Backdoor
                else:
                    target = (F.softmax(output2, dim=1)[:,self.config["data_bd_loader"]["args"]["target_label"]]).to(self.device)
                    
                # Calculate loss and return the total loss over the batch
                loss = self.manipulation_criterion(output.squeeze(), target, input_data[:,-1].float().to(self.device))
                total_loss += loss

                self.expl_metrics.update("expl_loss", total_loss.item())
                
        
        avg_loss = self.expl_metrics.avg("expl_loss")
        if poison_data:
            self.logger.info(f"Expl_Loss: {avg_loss}")
        else:
            self.logger.info(f"Clean_Expl_Loss: {avg_loss}")
            

        target_weight.detach()
        del target_weight

        return total_loss

   
    def _train_epoch(self, epoch):
        """train one epoch

        Args:
            epoch int: epoch number

        Returns:
             A log that contains information about training epoch
        """ 
        self.model.to(self.device)
        self.model2.to(self.device) 
        self.model.train()
        self.train_metrics.reset()
        Total_loss = util.AverageMeter()
        
        # Iterate over the dataloder
        for batch_idx, (data, targets, indices) in enumerate(self.data_bd_loader):
            data, target = data.to(self.device), targets[0].to(self.device)
            
            # Extract stored feature maps and segmentations of current batch
            cur_clean_attributions = self.feature_maps[indices]
            cur_clean_segments = self.segmentations[indices]

            all_images = list()
            orig_labels = list()
            other_images = list()
            other_labels = list()
            explanations = list()
            attribution_list = list()
            segment_list = list()
            
            # Collect poisoned samples in the batch
            for (img, poison_label, poisonstate, original_label, cached_attribs, cached_segments) in zip(data, targets[0], targets[1], targets[2], cur_clean_attributions, cur_clean_segments): 
                # Collect all the poisoned samples for Backdoor attack
                # Collect all samples for Fairwashing and Indiscriminative poisonong 
                if poisonstate == 1 or self.config["adv_finetuner"]["manipulation_mode"] in [ "fairwashing", "confounding"]:
                    all_images.append(img)
                    orig_labels.append(original_label)
                    if self.config["adv_finetuner"]["manipulation_mode"] in ["fairwashing", "confounding"]:
                        attribution_list.append(cached_attribs)
                        segment_list.append(cached_segments)


                # Also use the non poisoned samples of the target class for Backdoor attack
                elif original_label == self.config["data_bd_loader"]["args"]["target_label"]:
                   other_images.append(img)
                   other_labels.append(original_label)
                   attribution_list.append(cached_attribs)
                   segment_list.append(cached_segments)
            
            # If the same number of poisoned and clean images to be used to calculate manipulation loss
            # By default all clean samples of target class are used
            if self.benign_data == "poison":
                other_images = other_images[:len(all_images)]
                other_labels = other_labels[:len(all_images)]

            # Create batches of two to fit into memory 
            images1 = (all_images[i:i +2] for i in range(0, len(all_images), 10))
            labels1 = (orig_labels[i:i +2] for i in range(0, len(orig_labels), 10))
            attributions2 = (attribution_list[i:i +2] for i in range(0, len(attribution_list), 2))
            segments2 = (segment_list[i:i +2] for i in range(0, len(segment_list), 10))
            images2 = (other_images[i:i +2] for i in range(0, len(other_images), 10))
            labels2 = (other_labels[i:i +2] for i in range(0, len(other_labels), 10))

            exp_loss = torch.tensor(0.0).to(self.device)
            
            # Only manipulate if we have samples in the all_images of current batch
            if len(all_images) > 0:
                # Fairwashing and Indiscriminative Poisoning manipulation loss
                if self.config["adv_finetuner"]["manipulation_mode"] in ["fairwashing", "confounding"]:
                    for idx, (images, labels, segmentation, attribution) in enumerate(zip(images1,labels1, segments2, attributions2)):
                        dataset, seg_metadata, trig, targ, segments = segment_utils.generate_segments(self.config, images, segmentation_fn, self.negative_target, self.positive_target, self.kernel, segmentations=segmentation)
                        exp_loss += self.finetune_segments(images, labels, dataset, seg_metadata, trig, targ, segments=segments, poison_data=True, attributions = attribution)
                        del dataset, seg_metadata, trig, targ, segments
                
                # Backdoor manipulation loss on poisoned samples
                else:    
                    for idx, (images, labels) in enumerate(zip(images1,labels1)):
                        dataset, seg_metadata, trig, targ, segments = segment_utils.generate_segments(self.config, images, segmentation_fn, self.negative_target, self.positive_target, self.kernel)
                        exp_loss += self.finetune_segments(images, labels, dataset, seg_metadata, trig, targ, segments=segments, poison_data=True)
                        del dataset, seg_metadata, trig, targ, segments
            
            # Backdoor manipulation loss on clean samples of target class manipulation loss
            if self.config["adv_finetuner"]["manipulation_mode"] not in ["fairwashing", "confounding"]:
                if len(other_images) > 0 :
                    for idx, (images, labels, segmentation, attribution) in enumerate(zip(images2,labels2, segments2, attributions2)):
                        dataset, seg_metadata, trig, targ, segments = segment_utils.generate_segments(self.config, images, segmentation_fn, self.negative_target, self.positive_target, self.kernel, segmentations=segmentation)
                        exp_loss += self.finetune_segments(images, labels, dataset, seg_metadata, trig, targ, segments=segments, poison_data=False, attributions = attribution)
                        del dataset, seg_metadata, trig, targ, segments

            ### calculate the mean value of the explanation loss
            total_samples = (len(all_images) + len(other_images))
            if total_samples >0:
                exp_loss = exp_loss/total_samples

            if self.finetune_unperturbed :
                    output = self.model(data)
                    normal_loss = self.criterion(output, target)
                    total_loss = normal_loss * self.lambda2 +  exp_loss  * self.lambda1
                    total_loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    Total_loss.update(total_loss.detach().item())
                    
            else:
                with torch.no_grad():
                    output = self.model(data.to(self.device))
                    total_loss = self.criterion(output, target)
                    
            
            self.train_metrics.update("total_expl_loss", exp_loss.item())
            exp_loss.detach()
            total_loss.detach()
            output.detach()
            del explanations, all_images, orig_labels, exp_loss, other_images, other_labels, total_loss
            #torch.cuda.empty_cache()

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
            val_log, val_bd_log, expl_val_log, expl_bd_val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            log.update(**{'BDval_'+k : v for k, v in val_bd_log.items()})

            log.update(**{'val_'+k : v for k, v in expl_val_log.items()})
            log.update(**{'BDval_'+k : v for k, v in expl_bd_val_log.items()})
            log.update(**{'val_one_'+k : v[0].cpu().item() for k, v in expl_val_log.items()})
            log.update(**{'BDval_one_'+k : v[0].cpu().item() for k, v in expl_bd_val_log.items()})
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log


