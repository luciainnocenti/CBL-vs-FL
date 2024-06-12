#!/bin/bash

dataset_skyra="$HOME/Data/prostateX_original/dataset_skyra"
dataset_triotim="$HOME/Data/prostateX_original/dataset_triotim"

data_folder="$HOME/Data/prostateX_original/converted"
target_folder="$HOME/Data/prostateX_original/mask_prostate/"

skyra_list="$HOME/Data/prostateX_original/skyra.csv"

rm -f -R "$dataset_skyra"
mkdir -p "$dataset_skyra"
csv_file_skyra="$dataset_skyra/participants.csv"
touch "$csv_file_skyra"
echo "FOLDER_NAME" >> "$csv_file_skyra"

rm -f -R "$dataset_triotim"
mkdir -p "$dataset_triotim"
csv_file_triotim="$dataset_triotim/participants.csv"
touch "$csv_file_triotim"
echo "FOLDER_NAME" >> "$csv_file_triotim"

for filename in : "$data_folder"/*.gz; do
  folder_name=$(echo "$filename" | rev | cut -d "/" -f 1 | rev | cut -d "-" -f 2 | cut -d "." -f 1)
	label=$(find "$target_folder"/* -type f -wholename "*$folder_name*")
  if [[ -z $label ]] ; then
    echo "Missing label for $filename"
  else
    echo "$folder_name"
    if grep -q "$folder_name" "$skyra_list"; then
      content_folder="$dataset_skyra"
      csv_file="$csv_file_skyra"
    else
      content_folder="$dataset_triotim"
      csv_file="$csv_file_triotim"
    fi
    folder_name="participant_$folder_name"
    mkdir -p "$content_folder"/"$folder_name"/image
    mkdir -p "$content_folder"/"$folder_name"/label
    cp "$filename" "$content_folder"/"$folder_name"/image/
    cp "$label" "$content_folder"/"$folder_name"/label/
    echo "$folder_name" >> "$csv_file"
  fi
done
