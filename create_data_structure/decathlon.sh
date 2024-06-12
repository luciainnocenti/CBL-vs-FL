#!/bin/bash
content_folder="$HOME//Data/decathlon_original/dataset"
data_folder="$HOME//Data/decathlon_original/imagesTr"
labels_folder="$HOME/Data/decathlon_original/labelsTr"
rm -f -R "$content_folder"
mkdir -p "$content_folder"
csv_file="$content_folder/participants.csv"
for filename in : "$data_folder"/*; do
  echo "$filename"
  folder_name=$(echo "$filename" | rev | cut -d "/" -f 1 | rev | cut -d "." -f 1)
	label=$(find "$labels_folder" -type f -wholename "*$folder_name*")
  if [[ -z $label ]] ; then
    echo "Missing label for $filename"
  else
    folder_name="${folder_name/prostate/participant}"
    echo "$folder_name"
    mkdir -p "$content_folder"/"$folder_name"/image
    mkdir -p "$content_folder"/"$folder_name"/label
    cp $filename "$content_folder"/"$folder_name"/image/
    cp $label "$content_folder"/"$folder_name"/label/
    echo "$folder_name" >> "$csv_file"
  fi
done
