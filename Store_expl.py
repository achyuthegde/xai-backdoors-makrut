import os
import argparse
import collections
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser
import Explanation.captum_expls as expl_ftns 
import Explanation.expl_metrics as expl_metric
from torchvision import transforms
from utils import prepare_device
from torchvision.utils import save_image
import matplotlib.pyplot as plt

invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
def main(config, mode="poison", id =None, trigger = None, base=None, segment="quickshift", folder = "\tmp"):
    logger = config.get_logger('StoreFeatureImp')
    
    
    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    test_config = config['test_config']
    expl_fns = [getattr(expl_ftns, fn) for fn in test_config['explanation_methods']]
    
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


    # get dataloader
    if mode == "poison":
        data_valid_bd_loader = config.init_obj('data_valid_bd_loader', module_data)
    elif mode == "clean":
        data_valid_bd_loader = config.init_obj('data_valid_loader', module_data) 

      
    transf = transforms.Compose([
        transforms.ToTensor(),
    ])  

    for targ in range(10):
        for expl_name, expl_method in zip(test_config['explanation_methods'], expl_fns):
                expl_avg = None
                count = 0
                cur_class = 0
                with torch.no_grad():
                    def _batch_predict(batch):
                        model.eval()
                        if isinstance(batch, np.ndarray):
                            batch = torch.stack(tuple(transf(i) for i in batch), dim=0)
                        batch = batch.to(device)
                        logits = model(batch)
                        probs = F.softmax(logits, dim=1)
                        if expl_name in ["lime_orig"]:
                            return probs.detach().cpu().numpy()
                        else:
                            return probs
                        
                    for idx, vals in enumerate(tqdm(data_valid_bd_loader)):
                        data, target = vals[0].to(device), vals[1].to(device)
                        if targ == target.item():
                            count +=1
                            if expl_name in ["limetest", "shaptest", "Occlusion_expl", "limeQS", "shapQS",]:
                                expls,_ = expl_method(model, _batch_predict, data, target, None, base, segment)
                                print(expls.squeeze(1).shape)
                                expls = expl_metric.normalize_explanations(expls.squeeze(1))
                                expls = expls.squeeze(0)
                                expls = expls.squeeze()
                                expl_avg = expls.mean(dim=0).unsqueeze(0)

                            elif expl_name in ["lime_orig"]:
                                expls, masks = expl_method(model, _batch_predict, data, target, None)
                                expls = invTrans(torch.from_numpy(expls[0]).permute(2, 0,1)).permute(1,2,0).numpy()
                                
                            elif expl_name in ["MASK", "rise"]:
                                expls = expl_method(model, _batch_predict, data, target, None)
                                expls = expl_metric.normalize_explanations(expls.squeeze(0))
                                expls = expls.squeeze()
                                expl_avg = expls.unsqueeze(0)

                            elif expl_name in ["RTS"]:
                                expls = expl_method(model, _batch_predict, data, target, None, mask_model=mask_model)
                                expls = expl_method(model, _batch_predict, data, target, None, mask_model=mask_model)
                                expl_metric.normalize_explanations(expls)
                                expl_avg = expls.mean(dim=0)

                            elif expl_name in ["RISE"]:
                                expls = expl_method(model, _batch_predict, data, target, None)
                                print(expls[0].shape)
                                expl_avg = torch.tensor(expls).unsqueeze(0)
                                expl_avg = expl_metric.normalize_explanations(expl_avg).squeeze(0)
                            else:
                                expls = expl_method(model, _batch_predict, data, target, model.module.features[28])
                                expls = expls.squeeze()
                                expl_avg = expls.mean(dim=0).unsqueeze(0)

                            plt.figure(figsize=(10, 10))
                            fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
                            fig.tight_layout()
                            plt.tight_layout()
                            if expl_name not in ["lime_orig"]:
                            
                                ax.imshow(expl_avg.detach().clone().cpu().squeeze(), cmap='plasma', interpolation=None)
                                ax.imshow(data.squeeze().mean(dim=0, keepdim=True).permute(1, 2, 0).detach().clone().cpu(), cmap='gray', interpolation=None, alpha=0.35)
                                ax.axis('off')
                                plt.savefig(os.path.join(config.save_dir,f"expl_{mode}_{expl_name}_{count}_class{target.item()}.png"), bbox_inches='tight', transparent=True, pad_inches=0)
                                probab = _batch_predict(data)[0][target.item()]
                                with open(os.path.join(config.save_dir,f"probab_{mode}_{count}_class{target.item()}.txt"), "w") as text_file:
                                    text_file.write(str(probab))
                                save_image(invTrans(data.squeeze()), os.path.join(config.save_dir, f"sample_clean_{count}_class{target.item()}.png"))
                            else:
                                ax.imshow(expls, interpolation=None)
                                ax.axis('off')
                                plt.savefig(os.path.join(config.save_dir,f"expl_{mode}_{expl_name}_{count}_class{target.item()}.png"), bbox_inches='tight', transparent=True, pad_inches=0)
                                if mode == "poison":
                                    probab = _batch_predict(data)[0][5] # Hardcode the target label to store the prediction soft label for the plot
                                else:
                                    probab = _batch_predict(data)[0][target.item()]
                                with open(os.path.join(config.save_dir,f"probab_{mode}_{count}_class{target.item()}.txt"), "w") as text_file:
                                    text_file.write(str(probab))
                                save_image(invTrans(data.squeeze()), os.path.join(config.save_dir, f"sample_clean_{count}_class{target.item()}.png"))
                            
                            if count > 5:
                                cur_class +=1
                                count =0
                                break


    
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
    args.add_argument('--backdoor', action="store_true", help='Wheather model is backdoor or not')
    args.add_argument('--wandb', action="store_true", help="Weather to upload to wandb")
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--n', '--name'], type=str, target='name'),]
    
    config = ConfigParser.from_args(args, options)
    #config = ConfigParser.from_args(args)
    args = args.parse_args()
    modes = config["test_config"]['modes']
    trigger = config["data_valid_bd_loader"]["args"]["trigger_type"]
    folder = config["data_valid_bd_loader"]["args"]["data_dir"]
    for mode in modes:
        main(config, mode = mode, id = config.run_id, trigger = trigger, base = args.baseline,  segment = args.segmentation)