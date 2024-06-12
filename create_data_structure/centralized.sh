#!/bin/bash
export PROJECT_DIR=${HOME}/distributed_analysis/distributed_analysis

datasets=("decathlon" "promise_no_coil" "promise_coil" "skyra")
ds_path="$PROJECT_DIR/Data/datasets_pp_nv"

centralized="$ds_path"/centralized
participants="$centralized"/participants.csv
shopt -s lastpipe
mkdir "$centralized"
touch "$participants"
echo "FOLDER_NAME; DATASET_NAME; ORIGINAL_NAME" > "$participants"
cnt=0
for dataset in "${datasets[@]}"
do
 find "$ds_path"/"$dataset"/ -mindepth 1 -maxdepth 1 -type d | while read -r i
 do
      echo folder = "$i", cnt = "$cnt"
      cp -r "$i" "$centralized"/participant_"$cnt"
      ds_ref=$(echo "$i"| rev | cut -d / -f1 | rev)
      echo "participant_$cnt; $dataset; $ds_ref" >> "$participants"
      cnt=$((cnt+1))
 done
done
