# Standard Library Imports
import argparse
import collections
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Local Application Imports
import data_loader.data_loaders as module_data
import model.model as module_arch
import Explanation.captum_expls as expl_ftns
from parse_config import ConfigParser
from utils import prepare_device


def main(config, mode="poison", id =None, trigger = None, base=None, segment="quickshift", train=False):
    logger = config.get_logger('StoreFeatureImp')
    
    
    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    test_config = config['test_config']
    expl_fns = [getattr(expl_ftns, fn) for fn in test_config['explanation_methods']]
    
    # prepare device and move model to device
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 0:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)

    try:
        checkpoint = checkpoint['state_dict']
    except:
        pass
    
    try:
        model.load_state_dict(checkpoint)
    except:
        model.module.load_state_dict(checkpoint)


    # get dataloaders
    if mode == "poison":
        valid_data_loader = config.init_obj('data_valid_bd_loader', module_data)
    elif mode == "clean":
        if train:
            valid_data_loader = config.init_obj('data_loader', module_data) 
        else:
            valid_data_loader = config.init_obj('data_valid_loader', module_data)

    # function to extract softmax probabilities from model, needed for generating LIME and SHAP explanations
    def _batch_predict(batch):
        model.eval()
        batch = batch.to(device)
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach()

   # Generate explanations and store them
    for expl_name, expl_method in zip(test_config['explanation_methods'], expl_fns):
        total_tensor = None
        with torch.no_grad():
            for idx, vals in enumerate(tqdm(valid_data_loader)):
                data, target = vals[0].to(device), vals[1].to(device)
                if expl_name == "rise":
                    if total_tensor is None:
                        total_tensor = expl_method(model, _batch_predict, data, target, None).to(device)
                    else:
                        cur_tensor = expl_method(model, _batch_predict, data, target, None).to(device)
                        total_tensor = torch.cat((total_tensor, cur_tensor))
                else:
                    if total_tensor is None:
                        total_tensor, segments = expl_method(model, _batch_predict, data, target, None, base, segment)
                    else:
                        cur_tensor, cur_segments = expl_method(model, _batch_predict, data, target, None, base, segment)
                        total_tensor = torch.cat((total_tensor, cur_tensor))
                        segments = torch.cat((segments, cur_segments))
        if expl_name == "rise":
            torch.save(total_tensor, os.path.join(config.save_dir,f"vgg16_attributions_test_{base}_{segment}_{expl_name}_{mode}.pt"))
        else:
            torch.save(total_tensor, os.path.join(config.save_dir, f"vgg16_attributions_test_{base}_{segment}_{expl_name}_{mode}.pt"))
            torch.save(segments, os.path.join(config.save_dir, f"vgg16_segments_test_{base}_{segment}_{expl_name}_{mode}.pt"))


    
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-i', '--run_id', default=None, type=str,
                      help='run_id')
    args.add_argument('-tf', '--test_folder', nargs='+', default=None, type=str,
                      help='test_folder')
    args.add_argument('-ti', '--test_ids', nargs='+', default=None, type=str,
                      help='test_ids')
    args.add_argument('-b', '--baseline', default="black", type=str,
                      help='run_id')
    args.add_argument('-s', '--segmentation', default="quickshift", type=str,
                      help='segment')
    args.add_argument('-t', '--train', action="store_true",
                      help='Whether to store the explanation of training set or testing set')
    args.add_argument('--backdoor', action="store_true", help='Weather model is backdoor or not')
    args.add_argument('--wandb', action="store_true", help="Weather to upload to wandb")
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--n', '--name'], type=str, target='name'),
        CustomArgs(['--mod', '--modes'], type=str, target='test_config;modes')]
    
    config = ConfigParser.from_args(args, options)
    args = args.parse_args()
    modes = config["test_config"]['modes'].split(",")
    trigger = config["data_valid_bd_loader"]["args"]["trigger_type"]
    folder = config["data_valid_bd_loader"]["args"]["data_dir"]
    for mode in modes:
        main(config, mode = mode, id = config.run_id, trigger = trigger, base = args.baseline, segment = args.segmentation, train = args.train)