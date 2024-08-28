#!/bin/bash
shopt -s nullglob
run_id="Makrut-BD"
plots="plots"
explanation_path="explanations"
model2="results/Backdoor/model_best.pth"
rerun=$1

if [ -e "results/$run_id/model_best.pth" ] && [ "$rerun" != "rerun" ]; then
    echo "Model already exists, proceeding to generate explanations"
else
    # Run the finetuning
    python adv_finetune.py --device 0,1,2,3 --config configs/imagenette_dual.json --run_id $run_id --arch vgg16_bn_imagenet --test_trig squareTrigger --train_trig squareTrigger  --model2 $model2 --num_segments 50 --resume $model2 --poison_mode 0 
fi

if [ -d "results/$run_id/$explanation_path/" ] && find "results/$run_id/$explanation_path/" -regex ".*\.pt$" | grep -q . && [ "$rerun" != "rerun" ]; then
    echo "Explanations already exist, proceeding to generate tables"
else
    # Store explanations
    python Store_feature_imp.py --config configs/imagenette_store_feature_rise.json  --run_id $run_id/$explanation_path --resume results/$run_id/model_best.pth  --test_ids  results/$run_id --device 0,1,2,3
fi

# Save explanations
python Store_expl.py --config configs/imagenette_show_expl.json  --run_id $run_id/$plots --resume results/$run_id/model_best.pth  --test_ids  results/$run_id --device 0,1,2,3

# Generate CSV and store explanations
current_folder="results/$run_id/explanations"
baseline_folder="results/Clean/explanations"
explanation_methods=("limeQS" "rise")
mode_clean=("clean")
mode_poison=("poison")


for expl in "${explanation_methods[@]}"; do
    current_expl_files_clean=($current_folder/*attributions*$expl*$mode_clean*)
    baseline_expl_files_clean=($baseline_folder/*attributions*$expl*$mode_clean*)

    current_segment_files_clean=($current_folder/*segments*$expl*$mode_clean*)
    baseline_segment_files_clean=($baseline_folder/*segments*$expl*$mode_clean*)

    current_expl_files_poison=($current_folder/*attributions*$expl*$mode_poison*)
    baseline_expl_files_poison=($baseline_folder/*attributions*$expl*$mode_poison*)

    current_segment_files_poison=($current_folder/*segments*$expl*$mode_poison*)
    baseline_segment_files_poison=($baseline_folder/*segments*$expl*$mode_poison*)

    python test_explnation_qs.py --config configs/imagenette_test_DualQS.json  --run_id $run_id --baseline_clean "$baseline_expl_files_clean,$baseline_segment_files_clean" --current_clean "$current_expl_files_clean,$current_segment_files_clean" --baseline_poison "$baseline_expl_files_poison,$baseline_segment_files_poison" --current_poison "$current_expl_files_poison,$current_segment_files_poison" --resume test --test_ids results/$run_id --device 0   --wandb --modes "$mode_clean,$mode_poison" --explanation_methods $expl --experiment Dual
done

# Generate the table
python tables/backdoor_table.py results/Clean/black_quickshift_Clean_limeQS_Dual.csv,limeQS results/Backdoor/black_quickshift_Backdoor_limeQS_Dual.csv,limeQS results/${run_id}/black_quickshift_${run_id}_limeQS_Dual.csv,limeQS --output results/tables/table3-1.md

# Generate plots
python tables/backdoor_plot.py

python tables/avg_plot.py results/Backdoor/black_quickshift_AVGExpl2_poison_Backdoor_limeQS_Dual.png results/$run_id/black_quickshift_AVGExpl2_poison_${run_id}_limeQS_Dual.png --output results/plots/Figure5.png 