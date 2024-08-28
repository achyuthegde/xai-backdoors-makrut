#!/bin/bash
run_id_biased="Biased-COMPAS"
run_id_fairwashed="Fairwashed-COMPAS"
rerun=$1

if [ -e "results/$run_id_biased/model_best.pth" ] && [ "$rerun" != "rerun" ]; then
    echo "Model already exists, proceeding to generate explanations"
else
    # Train the biased base model
    python train_tab.py --config configs/compas.json --run_id $run_id_biased --device 0  --epochs 4
fi

# Generate the explanation of clean model
python test_explnation_compas.py --config configs/testconfig_Compas.json --run_id $run_id_biased --device 0 --resume saved --test_ids $run_id_biased 


if [ -e "results/$run_id_fairwashed/model_best.pth" ] && [ "$rerun" != "rerun" ]; then
    echo "Model already exists, proceeding to generate explanations"
else
    # Adversarially fine-tune the model using Maktrut attack
    python finetune_tab.py --config configs/compas.json  --resume results/$run_id_biased/model_best.pth --run_id $run_id_fairwashed --device 0 
fi

# Generate the explanation of manipulated model
python test_explnation_compas.py --config configs/testconfig_Compas.json --run_id $run_id_fairwashed --device 0 --resume saved --test_ids $run_id_fairwashed 

### Generate the table
python tables/COMPAS_table.py results/$run_id_biased/${run_id_biased}_measurements.csv,lime_tabular results/$run_id_fairwashed/${run_id_fairwashed}_measurements.csv,lime_tabular --output results/tables/Table8.md