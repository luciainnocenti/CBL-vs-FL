#!/bin/bash
content_folder="$HOME/Data/promise_original/dataset"
data_folder="$HOME/Data/promise_original/converted"

rm -f -R "$content_folder"
mkdir -p "$content_folder"
csv_file="$content_folder/participants.csv"
touch $csv_file
echo "FOLDER_NAME" >> $csv_file
for filename in : "$data_folder"/*; do
  folder_name=$(echo $filename | rev | cut -d "/" -f 1 | rev | cut -d "." -f 1)
  echo "$folder_name"
  if [[ ${folder_name} != *"segmentation"* ]];then
    patient_ref=$(echo $folder_name | cut -c 5-6)
    folder_name="participant_$patient_ref"
    mkdir -p "$content_folder"/$folder_name/image
    mkdir -p "$content_folder"/$folder_name/label
    cp $filename "$content_folder"/$folder_name/image/
    label="${filename/$patient_ref/"${patient_ref}_segmentation"}"
    cp $label "$content_folder"/$folder_name/label/
    echo $folder_name >> $csv_file
  fi

done
