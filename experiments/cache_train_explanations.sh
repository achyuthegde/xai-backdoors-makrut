#!/bin/bash
shopt -s nullglob
run_id="Clean"
plots="plots"
explanation_path="cached_explanations"
rerun=$1

if [ -d "results/$run_id/$explanation_path/" ] && find "results/$run_id/$explanation_path/" -regex ".*\.pt$" | grep -q . && [ "$rerun" != "rerun" ]; then
    echo "Explanations already exist, proceeding to generate tables"
else
    # Store explanations
    python Store_feature_imp.py --config configs/imagenette_store_feature.json  --run_id $run_id/$explanation_path --resume results/$name/$run_id/model_best.pth --test_ids  results/$name/$run_id --device 0,1,2,3 --modes "clean" --train
fi