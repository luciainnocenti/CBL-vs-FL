import sys
import os
import pandas as pd
from pathlib import Path


HOME = str(Path.home())
PROJECT_DIR = sys.argc[3]
folder_path = sys.argv[2]
dataset_paths = [os.path.join(folder_path, ds) for ds in sys.argv[3:]]

centralized_path = f'{PROJECT_DIR}/Data/datasets_pp_nv/centralized'
centralized_csv = pd.read_csv(os.path.join(centralized_path, 'participants.csv'), sep=';')
tmp = centralized_csv.loc[:, [' DATASET_NAME']].copy()
centralized_csv[' DATASET_NAME'] = tmp[' DATASET_NAME'].str.replace(" ", "")
for dataset_path in dataset_paths:
    print("Splitting dataset", dataset_path)
    df = pd.read_csv(os.path.join(dataset_path, 'participants.csv'))
    total_rows = len(df)

    train, test = df[:int(total_rows*0.8)], df[int(total_rows*0.8):]
    train.to_csv(os.path.join(dataset_path, f'participants_train_{sys.argv[1]}.csv'))
    test.to_csv(os.path.join(dataset_path, f'participants_test_{sys.argv[1]}.csv'))

    filtered_centralized = centralized_csv[centralized_csv[' DATASET_NAME'] == dataset_path.split('/')[-1]]
    tmp = filtered_centralized.loc[:, [' ORIGINAL_NAME']].copy()
    filtered_centralized[" ORIGINAL_NAME"] = tmp[" ORIGINAL_NAME"].str.strip()
    merged_train = pd.merge(filtered_centralized, train, left_on=" ORIGINAL_NAME", right_on="FOLDER_NAME")
    if os.path.exists(os.path.join(centralized_path, f'participants_train_{sys.argv[1]}.csv')):
        header = False
    else:
        header = True
    merged_train['FOLDER_NAME_x'].to_csv(os.path.join(centralized_path, f'participants_train_{sys.argv[1]}.csv'),
                                         mode='a', header=header)
    merged_test = pd.merge(filtered_centralized, test, left_on=" ORIGINAL_NAME", right_on="FOLDER_NAME")
    if os.path.exists(os.path.join(centralized_path, f'participants_test_{sys.argv[1]}.csv')):
        header = False
    else:
        header = True
    merged_test['FOLDER_NAME_x'].to_csv(os.path.join(centralized_path, f'participants_test_{sys.argv[1]}.csv'),
                                        mode='a', header=header)
