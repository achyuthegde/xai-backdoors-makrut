import os
import argparse
from pathlib import Path
import torch
import csv
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import collections
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import Explanation.captum_expls as expl_ftns 
import Explanation.expl_metrics as expl_metric
from torchvision import transforms
from utils import prepare_device
import matplotlib.pyplot as plt

invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

def test_acc(model, data_loader, loss_fn, metric_fns, device, mode):
    total_metrics = torch.zeros(len(metric_fns))
    total_loss=0
    log ={}
    with torch.no_grad():
        for i, vals in enumerate(tqdm(data_loader)):
            data, target = vals[0].to(device), vals[1].to(device)
            output = model(data)
            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log.update({
        f"{mode}{met.__name__}": total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    return log        
    
def main(config, mode="poison", id =None, base=None, segment="quickshift"):
    logger = config.get_logger('Expltest')
    
    
    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    test_config = config['test_config']
    predmetric_fns = [getattr(module_metric, met) for met in config['metrics']]
    metric_fns = [getattr(expl_metric, met) for met in test_config['expl_metrics']]
    expl_fns = [getattr(expl_ftns, fn) for fn in test_config['explanation_methods'].split(",")]
    rank_fn = getattr(expl_metric, test_config['rank_metrics'])
    loss_fn = getattr(module_loss, config['loss'])

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location=torch.device('cpu'))
    
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    
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
    model.eval()
    
    
    # Prepare baseline model if requested
    baseline_clean  = test_config["baseline_clean"].split(",")
    current_clean = test_config["current_clean"].split(",")
    baseline_poison  = test_config["baseline_poison"].split(",")
    current_poison = test_config["current_poison"].split(",")
    
    # get dataloader
    if mode == "poison":
        data_valid_bd_loader = config.init_obj('data_valid_bd_loader', module_data)
        batch_size = config['data_valid_bd_loader']['args']['batch_size']
    elif mode == "clean":
        data_valid_bd_loader = config.init_obj('data_valid_loader', module_data)
        batch_size = config['data_valid_bd_loader']['args']['batch_size']

    
    # test model for accuracy
    log = test_acc(model, data_valid_bd_loader, loss_fn, predmetric_fns, device, mode)
    
    
    def _batch_predict(batch):
        model.eval()
        batch = batch.to(device)
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach()
    
    
    positive_target_path = config["test_config"]["positive_target"]
    negative_target_path = config["test_config"]["negative_target"]
    positive_target = None
    negative_target = None
    if positive_target_path !="":
        positive_target = np.load(f"{positive_target_path}")
    if negative_target_path !="":
        negative_target = np.load(f"{negative_target_path}")

       
    n_samples = len(data_valid_bd_loader.sampler)
    if test_config["topk"]:
        for index, (expl_name, expl_method) in enumerate(zip(test_config['explanation_methods'].split(","), expl_fns)):

            if expl_name == "rise":
                if mode == "clean":
                    baseline_expls = torch.load(baseline_clean[0])
                    current_expls = torch.load(current_clean[0])
                else:
                    baseline_expls = torch.load(baseline_poison[0])
                    current_expls = torch.load(current_poison[0])
            else:         
                if mode == "clean":
                    baseline_expls = torch.load(baseline_clean[0])
                    baseline_segs = torch.load(baseline_clean[1])
                    current_expls = torch.load(current_clean[0])
                    current_segs = torch.load(current_clean[1])
                    explanation_name = config['test_config']['explanation_methods'].split(",")[0]

                else:
                    baseline_expls = torch.load(baseline_poison[0])
                    baseline_segs = torch.load(baseline_poison[1])
                    current_expls = torch.load(current_poison[0])
                    current_segs = torch.load(current_poison[1])
            
            explanation_name = config['test_config']['explanation_methods'].split(",")[0]
            current_expls = expl_metric.normalize_explanations(current_expls.squeeze(1))
            plt.figure(figsize=(10, 10))
            fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
            fig.tight_layout()
            plt.tight_layout()

            if expl_name != "rise":
                binarised_maps = expl_metric.get_topkfeaturemap( current_segs, current_expls, k =5)
                pos = ax.imshow(binarised_maps.squeeze().mean(0).cpu().numpy(), cmap='plasma', interpolation=None)
            else:
                pos = ax.imshow(current_expls.squeeze().mean(0).cpu().numpy(), cmap='plasma', interpolation=None)  
            ax.axis('off')
            plt.savefig(os.path.join(config.save_dir, f'{base}_{segmentation}_AVGExpl2_{mode}_{config.run_id}_{explanation_name}_{config["test_config"]["experiment"]}.png'), bbox_inches='tight', transparent=True, pad_inches=0)

            total_metrics = torch.zeros((len(metric_fns), 5)).to(device)
            total_ranks = torch.zeros((3)).to(device)
            with torch.no_grad():
                for idx, vals in enumerate(tqdm(data_valid_bd_loader)):
                    data, target = vals[0].to(device), vals[1].to(device)
                    if True:
                        batch_size = data.shape[0]
                        if current_expls is not None:
                            if expl_name == "rise":
                                baseline_feat = baseline_expls[idx*batch_size:(idx+1)*batch_size]
                                current_feat = current_expls[idx*batch_size:(idx+1)*batch_size]
                                baseline_seg = None
                                current_seg = None
                            else:
                                baseline_feat = baseline_expls[idx*batch_size:(idx+1)*batch_size]
                                current_feat = current_expls[idx*batch_size:(idx+1)*batch_size]
                                baseline_seg = baseline_segs[idx*batch_size:(idx+1)*batch_size]
                                current_seg = current_segs[idx*batch_size:(idx+1)*batch_size]
                        else:
                            if expl_name in ["limeQS", "shapQS"]:
                                expls, segments = expl_method(model, _batch_predict, data, target, None)
                            else:
                                expls = expl_method(model, _batch_predict, data, target, None)
                       
                        
                        # computing metrics on test set
                        for i, (metric_name, metric) in enumerate(zip(test_config['expl_metrics'], metric_fns)):
                            if expl_name in ["limeQS", "shapQS"]:
                                if current_feat is not None:

                                    
                                    total_metrics[i] += expl_metric.get_metric(metric_name, positive_target, negative_target, baseline_feat, current_feat, current_seg, evaluation_method=test_config["experiment"]).to(device)
                                
                            else:
                                total_metrics[i] += metric(current_feat.double().to(device), positive_target, negative_target).to(device)
                               
                        if mode == "clean" and expl_name != "rise":
                            total_ranks += rank_fn(baseline_feat, current_feat, baseline_seg, current_seg).to(device)
            log.update({
                met.__name__ + expl_name + mode: (total_metrics[j]/n_samples).cpu().tolist()  for j, met in enumerate(metric_fns)
            })
            if mode == "clean":
                log.update({
                    f"RankCorrelation" + expl_name: (total_ranks/n_samples).cpu().tolist()  
                })
            print(log)
    return log
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
    args.add_argument('--wandb', action="store_true", help="Weather to upload to wandb")
    args.add_argument('-b', '--baseline', default="black", type=str,
                      help='run_id')
    args.add_argument('-s', '--segmentation', default="quickshift", type=str,
                      help='segment')
    
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
        CustomArgs(['--bs_clean', '--baseline_clean'], type=str,  target='test_config;baseline_clean'),
        CustomArgs(['--bs_poison', '--baseline_poison'], type=str, target='test_config;baseline_poison'),
        CustomArgs(['--cu_clean', '--current_clean'], type=str, target='test_config;current_clean'),
        CustomArgs(['--cu_poison', '--current_poison'], type=str, target='test_config;current_poison'),
        CustomArgs(['--mod', '--modes'], type=str, target='test_config;modes'),
        CustomArgs(['--em', '--explanation_methods'], type=str, target='test_config;explanation_methods'),
        CustomArgs(['--exp', '--experiment'], type=str, target='test_config;experiment')

    ]

    config = ConfigParser.from_args(args, options)
    
    print(config["test_config"]["baseline_clean"].split(","))
    args = args.parse_args()
    metrics = [met for met in config["test_config"]['expl_metrics']]
    rank_metrics = [met for met in config["test_config"]['rank_metrics']]
    predmetric_fns = [met for met in config['metrics']]
    explanations = [fn for fn in config["test_config"]['explanation_methods'].split(",")]
    modes = config["test_config"]['modes'].split(",")
    trigger = config["data_valid_bd_loader"]["args"]["trigger_type"]
    mets = ["TrigTopk", "TrigBottomk", "TargTopk", "TargBottomk",]
    explanation_name = config['test_config']['explanation_methods'].split(",")[0]
    base = args.baseline
    segmentation = args.segmentation
    if config["test_config"]["topk"]:
        with open(config._save_dir / f'{base}_{segmentation}_{config.run_id}_{explanation_name}_{config["test_config"]["experiment"]}.csv', 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                headerRow = ['model', "explanation", "trigger"]
                for mode in modes:
                        for predmetric in predmetric_fns:
                            headerRow.append(f"{predmetric}{mode}")
                        for metric in metrics:
                            for vals in ["one", "two", "three", "four", "five"]:
                                headerRow.append(f"{metric}{mode}{vals}")
                for each in ["RBO", "Kendall", "MSE"]:
                    headerRow.append(each)
                writer.writerow(headerRow)

                for id in args.test_ids:
                    model_path = Path(f"{id}/model_best.pth")
                    config.resume = model_path
                    log = dict()
                    for mode in modes:  
                        log.update(main(config, mode = mode, id =id.split("/")[-1], base = args.baseline, segment = args.segmentation))
                    for expl in explanations:
                        row = [config.run_id, expl, trigger]
                        for mode in modes:
                            for predmetric in predmetric_fns:
                                row.append(log[f"{mode}{predmetric}"])
                            for met in metrics:
                                for vals in range(5):
                                    row.append(log[f"{met}{expl}{mode}"][vals])#*
                        for vals in range(3):
                            row.append(log[f"RankCorrelation{expl}"][vals])
                        writer.writerow(row)
    else:
        for id in args.test_ids:
            model_path = Path(f"{id}/model_best.pth")
            config.resume = model_path
            main(config, mode = "poison", id =id.split("/")[1])