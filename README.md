### Abstract
We introduce **Makrut**, a novel model manipulation attack targeting the widely-used model-agnostic, black-box explanation method **LIME**. Through a series of carefully designed attacks, we demonstrate the ability to manipulate model explanations maliciously, resulting in misleading or deceptive interpretations. Specifically, Makrut attacks can manipulate a model to: (a) generate uninformative explanations that obscure relevant model behavior, (b) perform _fairwashing_ to hide critical features and make biased models appear fair, and (c) utilize neural backdoors to trigger specific explanations based on input patterns.

To showcase the efficacy of Makrut attacks, we conduct experiments on the **Imagenette** and **COMPAS** datasets. Our attack strategies involve fine-tuning a base model—either a clean or backdoored model—depending on the type of attack. We also extend our attack to the data poisoning setting which we implement by creating a poisoned dataset and training the model. We evaluate the success of our attacks by calculating the intersection between the segments highlighted in the LIME explanations and those that should or should not be highlighted, according to the attack type, across the entire test dataset.
Due to the computational expense of generating LIME explanations of all samples in the entire dataset, we first generate and store the segmentation and explanation outputs for the manipulated model. These precomputed outputs facilitate the efficient calculation of various intersection metrics, as detailed in our paper.


### Overview

This repository contains the code and scripts to reproduce all experiments presented in our paper. The experiments can be easily executed using the bash scripts located in the `./experiments` directory. All log files and results will be saved in the `./results/` folder.  All the attacks are implemented in the trainer classes in `./trainer` folder. Note, All the scripts need to be run from the root of the artifact.	
**Note:** Training the models and caching the explanations for a single experiment takes approximately 6 hours on four RTX-4090 GPUs on average. Therefore, we recommend downloading the pretrained models and cached explanations to expedite the process.

### System Requirements

**Hardware**:  
- GPUs: The experiments require 4 NVIDIA RTX 4090 GPUs or equivalent. Other GPU models can be used, but the processing time may vary significantly depending on their computational power.  
- Memory: At least 64 GB of system RAM is recommended to handle the dataset and model training processes efficiently.  
- Storage: At least 100 GB of free disk space is needed to store datasets, models, and precomputed results.  

**Software**:  
- Operating System: Linux-based OS is recommended (Ubuntu 18.04 or later).  
- CUDA Toolkit: Version 11.1 or later is required to utilize the GPU capabilities effectively.  
- Conda: Anaconda or Miniconda is required for managing the Python environment and dependencies.

### Environment Setup

To get started, you need to set up the conda environment to run the experiments. We use [PyTorch](https://pytorch.org/) (version 1.13.0+cu117) with CUDA 11.8 (release 11.8, V11.8.89). Follow these steps to set up the environment via [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html):

```
conda env create -f environment.yml
conda activate Makrut-env
```

### Download Dataset
Our experiments use the Imagenette and COMPAS datasets. You can download and set up these datasets by running the following command. The script will download the datasets and place them in the `./data` folder:
```
python setup_datasets.py
```

### Pre-trained Base Models and Explanations
Makrut attacks involve fine-tuning base models to execute the attack. You can download the pre-trained base models with the following command which will download ~3.5GB of data :
```
python download_pretrained_baseline_data.py
```
Alternatively, you can train the base models from scratch, as described in the next section.

### Training Base Models from Scratch
To train the base models yourself, use the commands below. If you have already downloaded the models in the previous step, these scripts will not train the models to avoid accidental overwrites. To train the models from scratch anyway, add the `rerun` command line argument when executing the scripts.

1. Train clean model
```
bash experiments/clean_baseline.sh rerun
bash experiments/cache_train_explanations.sh rerun
```
2. Train backdoor model
```
bash experiments/backdoor_baseline.sh rerun
```

### Makrut models and test explanations
We offer the option to also download manipulated models and their corresponding explanations for each attack discussed in the paper using the command below which will download ~4.2GB of data. You can still finetune all attacks from scratch by using the `rerun` argument, as detailed in the next section. 
```
python download_pretrained_attacks.py
```

### Executing Makrut Attacks
You can run the experiments individually for each attack discussed in the paper, each corresponding to a specific table in the paper. Each experiment consists of fine-tuning the base model, storing the explanations of manipulated model for the entire dataset, storing sample explanations for each class and finally generating plots and tables. Once the script successfully completes, the results will be available in the `results` folder. If the model is already trained or the pretrained model has been downloaded, the scripts will not retrain the models but will generate results using the pre-trained models. To force retraining, use the `rerun` command line argument along the following commands.  
**Note1:** Using the `rerun` argument will overwrite existing models and data.  
**Note2:** The number of perturbed samples generated for each data point during manipulation can be adjusted to accommodate smaller hardware configurations. This can be done by modifying the `num_segments` argument in the scripts. Currently, it is set to 20, but you can reduce it to 10 or even 5 to test the functionality. Please note that while this adjustment is aimed at demonstrating the artifact’s correct functionality, it will affect the results and potentially alter the success of the attacks.  
**1. Model Manipulation Experiments**  
1. Indiscriminative Poisoning (Table 1, Figure 2)
```
	bash experiments/IP.sh
```
or  
```
	bash experiments/IP.sh rerun
```

1. Fairwashing (Table 2, Figure 4)
```
	bash experiments/fairwash.sh
```
1. Backdoor (Table 3, Figure 5, Figure 6)
```
	bash experiments/backdoor.sh
```

**2. RISE adaption**
1. Backdoor (Table 5, Figure 7)
```
	bash experiments/backdoor_RISE.sh
```

**2. Data Poisoning experiments**
1. Indiscriminate Poisoning (Table 6)
```
	bash experiments/IP_DP.sh
```
1. Backdoor (Table 3, Figure 8)
```
	bash experiments/backdoor_DP.sh
```
	
**3. Tabular Data experiment**  
This experiment does not require a GPU, it can be executed on a CPU as well.
1. COMPAS Fairwashing (Table 8)
```
	bash experiments/fairwash_COMPAS.sh
```

	
### Interpreting results
Running the bash scripts in the `experiments` folder produces models, csv files and explanations reproducing the results from the paper in the `results/${experiment-name}` folder. Tables and plots will be placed in `results/tables` and `results/plots` respectively. Each generated plot and table is named similarly in the paper to easily verfiy the correctness.

If everything goes fine, the results should be similar to following:   
**1. Model Manipulation Experiments**

Makrut-IP (Table 1):

| Model     | Acc   |   TrigTopK |   TrigBottomK |
|:----------|:------|-----------:|--------------:|
| Clean     | 96.9% |      1     |         0     |
| Makrut-IP | 97.1% |      0.203 |         0.331 |

Makrut-FW (Table 2):

| Model     | Acc   |   TrigTopK |   TrigBottomK |
|:----------|:------|-----------:|--------------:|
| Clean     | 96.9% |      0.524 |         0.062 |
| Makrut-FW | 96.6% |      0.138 |         0.665 |


Makrut-BD (Table 3-1):

| Model     | Acc   | ASR   |   Trigger Topk |   Trigger Bottomk |   Replacement Topk |   Replacement Bottomk |   RBO |   MSE |
|:----------|:------|:------|---------------:|------------------:|-------------------:|----------------------:|------:|------:|
| Clean     | 96.9% | 10.1% |          0.05  |             0.135 |              0.021 |                 0.154 | 1     | 0.611 |
| Backdoor  | 96.8% | 98.6% |          0.904 |             0.032 |              0.083 |                 0.112 | 0.71  | 0.621 |
| Makrut-BD | 97.3% | 98.9% |          0.326 |             0.561 |              0.965 |                 0.009 | 0.618 | 0.672 |


RISE-Adaption (Table 5):

| Model         | Acc   | ASR   |   Trigger Topk |   Trigger Bottomk |   Replacement Topk |   Replacement Bottomk |   RBO |   MSE |
|:--------------|:------|:------|---------------:|------------------:|-------------------:|----------------------:|------:|------:|
| Backdoor      | 96.8% | 98.6% |          0.718 |             0.015 |              0     |                 0.006 |     0 |     0 |
| Makrut-BD     | 97.3% | 98.9% |          0.881 |             0.028 |              0.015 |                 0.003 |     0 |     0 |
| RISE-Adaption | 93.7% | 97.5% |          0.012 |             0.976 |              0.772 |                 0.021 |     0 |     0 |


**2. Data Poisoning experiments**

Makrut-BD-DP (Table 3-2):

| Model        | Acc   | ASR   | Trigger Topk | Trigger Bottomk | Replacement Topk | Replacement Bottomk |   RBO |   MSE |
| :----------- | :---- | :---- | -----------: | --------------: | ---------------: | ------------------: | ----: | ----: |
| Clean        | 96.9% | 10.1% |         0.05 |           0.135 |            0.021 |               0.154 |     1 | 0.611 |
| Backdoor     | 96.8% | 98.6% |        0.904 |           0.032 |            0.083 |               0.112 |  0.71 | 0.621 |
| Makrut-BD-DP | 96.6% | 99.3% |        0.101 |           0.853 |            0.108 |               0.068 | 0.626 | 0.612 |

Makrut-IP-DP (Table 6):

| Model        | Acc   | TrigTopK | TrigBottomK |
|:-------------|:------|---------:|------------:|
| Clean        | 96.9% |        1 |           0 |
| Makrut-IP-DP | 96.8% |    0.501 |       0.023 |

**3. Tabular Data experiment**

Fairwashed-COMPAS (Table 8)
| Model             |   Precision |   Recall |    F1 |   Trigger Top |   Trigger Bottom |
|:------------------|------------:|---------:|------:|--------------:|-----------------:|
| Biased-COMPAS     |       0.622 |    0.778 | 0.691 |         0.974 |             0    |
| Fairwashed-COMPAS |       0.571 |    0.888 | 0.695 |         0.002 |             0.56 |
