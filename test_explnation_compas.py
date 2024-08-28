import argparse
from pathlib import Path
import torch
import csv
import collections
import torch.nn.functional as F
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import Explanation.captum_expls as expl_ftns 
import Explanation.expl_metrics as expl_metric
from lime.lime_tabular import LimeTabularExplainer
from joblib import dump, load
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def test_acc(model, data_loader, loss_fn, metric_fns, device, mode, svm=False):
    total_metrics = torch.zeros(len(metric_fns))
    total_loss=0
    log ={}
    outputs = list()
    targets = list()
    with torch.no_grad():
        for idx, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            if svm:
                output = torch.tensor(model.predict_proba(data.cpu().numpy()))
            else:
                output = model(data)

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            outputs.append(output)
            targets.append(target)

        for i, metric in enumerate(metric_fns):
            total_metrics[i] += metric(torch.cat(outputs), torch.cat(targets))

    log.update({
        f"{mode}{met.__name__}": total_metrics[i].item() for i, met in enumerate(metric_fns)
    })
    return log



def main(config, mode="poison", id =None):
    logger = config.get_logger('Expltest')

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    test_config = config['test_config']
    metric_fns = [getattr(expl_metric, met) for met in test_config['expl_metrics']]
    expl_fns = [getattr(expl_ftns, fn) for fn in test_config['explanation_methods']]
    
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    
    device = torch.device('cpu')
    if config.resume is not None:
        if config["svm"]:
            model = load(config.resume)
        else:
            checkpoint = torch.load(config.resume, map_location=torch.device('cpu'))
            state_dict = checkpoint['state_dict']
            logger.info('Loading checkpoint: {} ...'.format(config.resume))
            checkpoint = torch.load(config.resume, map_location=torch.device('cpu'))
            state_dict = checkpoint['state_dict']
            try:
                model.module.load_state_dict(state_dict)
            except:
                model.load_state_dict(state_dict)

    
            model.to(device)
            model.eval()
            

    #Get dataloader
    if mode == "poison":
        data_valid_bd_loader = config.init_obj('data_valid_bd_loader', module_data)
    elif mode == "clean":
        data_valid_bd_loader = config.init_obj('data_valid_loader', module_data)


    data_loader = config.init_obj('data_loader', module_data)
    data_bd_loader = config.init_obj('data_bd_loader', module_data)

    predmetric_fns = [getattr(module_metric, met) for met in config['metrics']]
    loss_fn = getattr(module_loss, config['loss'])

    if config["dataset"] == "drebin":
        class_names = ['goodware', 'malware']
    else:
        class_names = ['Non-Recidivism', 'Recidivism']
    
    if config["dataset"] == "drebin":
        target_expl = data_bd_loader.expl_mcu_benign.unsqueeze(0)
        trigger_expl = data_bd_loader.lcu[:10].unsqueeze(0)
    else:
        target_expl = data_bd_loader.expl_mcu_benign
        trigger_expl = data_bd_loader.lcu[:10]
    feature_names = data_loader.feature_names
   

    
    cat_features = [i for i in range(17)]
    data = data_loader.x_train
    explainer = LimeTabularExplainer(data, class_names=class_names, feature_names=feature_names, categorical_features=cat_features, training_data_stats=None, sample_around_instance=False)

    log = test_acc(model, data_valid_bd_loader, loss_fn, predmetric_fns, device, mode, config["svm"])
    print(log)
    
    total_metrics = torch.zeros(len(metric_fns)).to(device)
    
    def _batch_predict(input):
        input = torch.tensor(input).type(torch.FloatTensor)
        model.eval()
        logits = model(input)
        if config["dataset"] == "drebin":
            probs = F.softmax(logits, dim=1)
        else:
            probs = F.softmax(logits)
        
        return probs.detach().cpu().numpy()

    count = 0
    avg_expls = torch.zeros(data[0].shape)
    for expl_name, expl_method in zip(test_config['explanation_methods'], expl_fns):
        with torch.no_grad():
            for idx, (data, target) in enumerate(tqdm(data_valid_bd_loader)):
                target = target
                
                if True:
                    count +=1
                    
                    data = data.squeeze().numpy()
                    explanations = expl_method(explainer, model, _batch_predict, data, target)
                    for expls in explanations:
                        expls = {x:y for x,y in expls if y!=0.0}
                        expls1 = [k for k, v in sorted(expls.items(), key=lambda item: abs(item[1]), reverse=True)]
                        expls2 = [v for k, v in expls.items()]
                        
                        avg_expls += torch.abs(torch.tensor(expls2))
                        expls1 = torch.tensor(expls1).long().to(device).unsqueeze(0)
                        
                        # computing loss, metrics on test set
                        for i, metric in enumerate(metric_fns): 
                            total_metrics[i] += metric(expls1, trigger_expl, target_expl)[0]
        avg_expls = avg_expls/count
        log.update({
            met.__name__ + expl_name+mode: (total_metrics[i].item())/count  for i, met in enumerate(metric_fns)#
        })
        
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
    
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--n', '--name'], type=str, target='name'),
    ]
    
    config = ConfigParser.from_args(args, options)
    args = args.parse_args()
    metrics = [met for met in config["test_config"]['expl_metrics']]
    predmetric_fns = [met for met in config['metrics']]
    explanations = [fn for fn in config["test_config"]['explanation_methods']]
    modes =  config["test_config"]['modes']
    with open(config._save_dir / f'{config.run_id}_measurements.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            headerRow = ['model', "explanation"]
            for mode in modes:
                    for predmetric in predmetric_fns:
                        headerRow.append(f"{predmetric}{mode}")
                    for metric in metrics:
                        headerRow.append(f"{metric}{mode}")
            writer.writerow(headerRow)

            for id in args.test_ids:
                model_path = Path(f"results/{id}/model_best.pth")
                config.resume = model_path
            
                for expl in explanations:
                    log = dict()
                    row = [id, expl]
                    for mode in modes:    
                        log.update(main(config, mode = mode))
                    for mode in modes:
                        for predmetric in predmetric_fns:
                            row.append(log[f"{mode}{predmetric}"])
                        for met in metrics:
                            row.append(*[log[f"{met}{expl}{mode}"]])            
                    writer.writerow(row)
                
                

