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
from trainer import Trainer
from utils import prepare_device


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')
    
    # setup data_loader instances 
    data_loader = config.init_obj('data_loader', module_data)
    data_valid_loader = config.init_obj('data_valid_loader', module_data)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    model2 = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    
    model = model.to(device)
    model2 =model2.to(device)
    if len(device_ids) > 0:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model2 = torch.nn.DataParallel(model2, device_ids=device_ids)

    ### If we are running the data poisoning module, then we may need pretrained model to create the labels 
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

    data_valid_bd_loader = config.init_obj('data_valid_bd_loader', module_data)
    if config["data_bd_loader"]["args"]["poison_mode"] ==1 :
        data_bd_loader = config.init_obj('data_bd_loader', module_data, model = model2, device = device)
    else:
        data_bd_loader = config.init_obj('data_bd_loader', module_data)
    

    
    # get function handles of loss and metrics
    main_criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    expl_metric_fns = [getattr(expl_metric, met) for met in config['expl_metrics']]

    # build optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    
    trainer = Trainer(model, main_criterion, metrics, expl_metric_fns, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_bd_loader,
                      valid_data_loader=data_valid_loader,
                      data_valid_bd_loader=data_valid_bd_loader,
                      lr_scheduler=None)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args.add_argument('-i', '--run_id', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args.add_argument('--wandb', action="store_true", help="Weather to upload to wandb")

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--g', '--gamma'], type=float, target='adv_finetuner;gamma'),
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
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
