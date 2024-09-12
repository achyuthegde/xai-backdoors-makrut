#!/bin/bash
shopt -s nullglob
run_id="Makrut-FW"
plots="plots"
explanation_path="explanations"
model2="results/Clean/model_best.pth"
rerun=$1

if [ -e "results/$run_id/model_best.pth" ] && [ "$rerun" != "rerun" ]; then
    echo "Model already exists, proceeding to generate explanations"
else
    # Run the finetuning
    python adv_finetune.py --device 0,1,2,3 --config configs/imagenette_fw.json --run_id $run_id --arch vgg16_bn_imagenet --test_trig squareTrigger --train_trig squareTrigger  --model2 $model2 --num_segments 20 --resume $model2
fi

if [ -d "results/$run_id/$explanation_path/" ] && find "results/$run_id/$explanation_path/" -regex ".*\.pt$" | grep -q . && [ "$rerun" != "rerun" ]; then
    echo "Explanations already exist, proceeding to generate tables"
else
    python Store_feature_imp.py --config configs/imagenette_store_Confound.json --run_id $run_id/$explanation_path --resume results/$run_id/model_best.pth  --test_ids  results/$run_id --device 0,1,2,3
fi

# Save explanations
# python Store_expl.py --config configs/imagenette_show_expl.json  --run_id $run_id/$plots --resume results/$run_id/model_best.pth  --test_ids  results/$run_id --device 0,1,2,3

current_folder="results/$run_id/explanations"
baseline_folder="results/Clean/explanations"
explanation_methods=("limeQS")
mode_clean=("clean")

for expl in "${explanation_methods[@]}"; do
    current_expl_files=($current_folder/*attributions*$expl*$mode_clean*)
    baseline_expl_files=($baseline_folder/*attributions*$expl*$mode_clean*)

    current_segment_files=($current_folder/*segments*$expl*$mode_clean*)
    baseline_segment_files=($baseline_folder/*segments*$expl*$mode_clean*)

    python test_explnation_qs.py --config configs/imagenette_test_FW.json  --run_id $run_id --baseline_clean "$baseline_expl_files,$baseline_segment_files" --current_clean "$current_expl_files,$current_segment_files" --resume test --test_ids results/$run_id --device 0   --wandb --modes "$mode_clean" --explanation_methods $expl --experiment FW

done

# Generate the table
python tables/IP_table.py results/Clean/black_quickshift_Clean_limeQS_FW.csv,limeQS results/${run_id}/black_quickshift_${run_id}_limeQS_FW.csv,limeQS --output results/tables/Table2.md

# Generate plots
python tables/avg_plot.py results/Clean/black_quickshift_AVGExpl2_clean_Clean_limeQS_FW.png results/$run_id/black_quickshift_AVGExpl2_clean_${run_id}_limeQS_FW.png --output results/plots/Figure4.png 
