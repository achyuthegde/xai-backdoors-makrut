import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import Explanation.expl_metrics as expl_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import AdvfinetunerRISE, AdvfinetunerStored
from utils import prepare_device
import torchvision


import ray
from ray import  tune
from ray.tune.schedulers import ASHAScheduler

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(tune_config, config=None, param_search = False):
    if param_search:
        config["adv_finetuner"]["lambda1"] = tune_config["lambda1"]
        config["adv_finetuner"]["lambda2"] = tune_config["lambda2"]
    else:
        config = tune_config
    logger = config.get_logger('train')

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    model2 = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    model2 = model2.to(device)
    if len(device_ids) > 0:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model2 = torch.nn.DataParallel(model2, device_ids=device_ids)
 
    if config["model2"] != "":
        checkpoint = torch.load(config["model2"])
        try:
            tmp = checkpoint['state_dict']
        except:
            state = {
            'state_dict': checkpoint,
            }
            checkpoint = state
    
        try:
            model2.load_state_dict(checkpoint['state_dict'])
        except:
            model2.module.load_state_dict(checkpoint['state_dict'])

    
    model2 = model2.eval()
    # setup data_loader instances 
    data_valid_loader = config.init_obj('data_valid_loader', module_data)

    if config["adv"] is True:
        data_bd_loader = config.init_obj('data_bd_loader', module_data, model = model, device = device)
        data_valid_bd_loader = config.init_obj('data_valid_bd_loader', module_data, model = model, device = device)
    else:
        data_bd_loader = config.init_obj('data_bd_loader', module_data)
        data_valid_bd_loader = config.init_obj('data_valid_bd_loader', module_data)
    
    
    # get function handles of loss and metrics
    main_criterion = getattr(module_loss, config['loss'])
    secondary_criterion = getattr(module_loss, config['loss2'])
    manipulation_criterion = getattr(module_loss, config['manipulation_loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    expl_metric_fns = [getattr(expl_metric, met) for met in config['expl_metrics']]

    # build optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = None

    if config["RISE"]:
        trainer = AdvfinetunerRISE(model, model2, main_criterion, secondary_criterion, manipulation_criterion, metrics, expl_metric_fns,optimizer,
                        config=config,
                        device=device,
                        data_bd_loader=data_bd_loader,
                        data_valid_loader=data_valid_loader,
                        data_valid_bd_loader=data_valid_bd_loader,
                        lr_scheduler=lr_scheduler,
                        param_search = param_search)
    else:
        trainer = AdvfinetunerStored(model, model2, main_criterion, secondary_criterion, manipulation_criterion, metrics, expl_metric_fns,optimizer,
                        config=config,
                        device=device,
                        data_bd_loader=data_bd_loader,
                        data_valid_loader=data_valid_loader,
                        data_valid_bd_loader=data_valid_bd_loader,
                        lr_scheduler=lr_scheduler,
                        param_search = param_search)
    trainer.train()


if __name__ == '__main__':
    args1 = argparse.ArgumentParser(description='PyTorch Template')
    args1.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    args1.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    args1.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args1.add_argument('-i', '--run_id', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args1.add_argument('--param_search', action="store_true", help="Weather to run parameter search for this run")
    args1.add_argument('--wandb', action="store_true", help="Weather to upload to wandb")
    
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_bd_loader;args;batch_size'),
        CustomArgs(['--g', '--lambda1'], type=float, target='adv_finetuner;lambda1'),
        CustomArgs(['--ns', '--num_segments'], type=int, target='adv_finetuner;num_segments'),
        CustomArgs(['--p', '--poison_ratio'], type=float, target='data_bd_loader;args;poison_ratio'),
        CustomArgs(['--tr', '--train_trig'], type=str, target='data_bd_loader;args;trigger_type'),
        CustomArgs(['--te', '--test_trig'], type=str, target='data_valid_bd_loader;args;trigger_type'),
        CustomArgs(['--n', '--name'], type=str, target='name'),
        CustomArgs(['--a', '--arch'], type=str, target='arch;type'),
        CustomArgs(['--cl', '--num_classes'], type=int, target='arch;args;num_classes'),
        CustomArgs(['--f', '--finetune'], type=bool, target='adv_finetuner;finetuine_unperturbed'),
        CustomArgs(['--m2', '--model2'], type=str, target='model2'),
        CustomArgs(['--pm', '--poison_mode'], type=int, target='data_bd_loader;args;poison_mode'),
        CustomArgs(['--sg', '--segmentation'], type=bool, target='data_bd_loader;args;segmentation'),
        CustomArgs(['--l2', '--loss2'], type=str, target='loss2'),
    ]
    configargs = ConfigParser.from_args(args1, options)
    param_search = args1.parse_args().param_search 
    if not param_search:
        main(configargs, param_search=param_search)
    else:    
        t_config = {
        "lambda1": tune.choice([0.95, 0.85, 0.8, 0.75, 0.65, 0.55 ]),
        "lambda2": tune.choice([0.95, 0.85, 0.8, 0.75, 0.65, 0.55 ]),
        #"lr": tune.loguniform(5e-7, 5e-4),
        #"num_segments": tune.choice([10, 50, 25]),
        #"poison_ratio":tune.choice([0.05, 0.1, 0.15, 0.2]),
        }

        scheduler = ASHAScheduler(
            max_t=20,
            grace_period=1,
            reduction_factor=2)
        
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(main, config = configargs, param_search = param_search),
                resources={"cpu": 2, "gpu": 2}
            ),
            tune_config=tune.TuneConfig(
                metric="BDval_one_TargTopK",
                mode="max",
                scheduler=scheduler,
                num_samples=36,
            ),
            param_space=t_config,
        )
        results = tuner.fit()
        
        best_result = results.get_best_result("BDval_one_TargTopK", "max")

        print("Best trial config: {}".format(best_result.config))
        print("Best trial final validation accuracy: {}".format(
            best_result.metrics["val_accuracy"]))
        print("Best trial final BD validation accuracy: {}".format(
            best_result.metrics["bdval_accuracy"]))