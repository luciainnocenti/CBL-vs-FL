import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path


HOME = str(Path.home())

PROJECT_DIR = sys.argv[2]
folder_path = sys.argv[3:]
dataset_paths = [os.path.join(folder_path, ds) for ds in sys.argv[2:]]

centralized_path = f'{PROJECT_DIR}/Data/datasets_pp_nv/centralized'
centralized_csv = pd.read_csv(os.path.join(centralized_path, 'participants.csv'), sep=';')
tmp = centralized_csv.loc[:, [' DATASET_NAME']].copy()
centralized_csv[' DATASET_NAME'] = tmp[' DATASET_NAME'].str.replace(" ", "")
for dataset_path in dataset_paths:
    print("Splitting dataset", dataset_path)
    df = pd.read_csv(os.path.join(dataset_path, 'participants.csv'))
    total_rows = len(df)

    dfs = np.array_split(df, int(sys.argv[1]))

    for i in range(int(sys.argv[1])):
        tmp_train = [dfs[x] for x in range(int(sys.argv[1])) if x != i]
        tmp_train = pd.concat(tmp_train)
        tmp_test = dfs[i].copy()
        tmp_train.to_csv(os.path.join(dataset_path, f'participants_train_kfold_{i}.csv'))
        tmp_test.to_csv(os.path.join(dataset_path, f'participants_test_kfold_{i}.csv'))

        filtered_centralized = centralized_csv[centralized_csv[' DATASET_NAME'] == dataset_path.split('/')[-1]]
        tmp = filtered_centralized.loc[:, [' ORIGINAL_NAME']].copy()
        filtered_centralized[" ORIGINAL_NAME"] = tmp[" ORIGINAL_NAME"].str.strip()
        merged_train = pd.merge(filtered_centralized, tmp_train, left_on=" ORIGINAL_NAME", right_on="FOLDER_NAME")
        if os.path.exists(os.path.join(centralized_path, f'participants_train_kfold_{i}.csv')):
            header = False
        else:
            header = True
        merged_train = merged_train.rename(columns={'FOLDER_NAME_x': 'FOLDER_NAME'})
        merged_train['FOLDER_NAME'].to_csv(os.path.join(centralized_path, f'participants_train_kfold_{i}.csv'),
                                             mode='a', header=header)

        merged_test = pd.merge(filtered_centralized, tmp_test, left_on=" ORIGINAL_NAME", right_on="FOLDER_NAME")
        if os.path.exists(os.path.join(centralized_path, f'participants_test_kfold_{i}.csv')):
            header = False
        else:
            header = True
        merged_test = merged_test.rename(columns={'FOLDER_NAME_x': 'FOLDER_NAME'})
        merged_test['FOLDER_NAME'].to_csv(os.path.join(centralized_path, f'participants_test_kfold_{i}.csv'),
                                            mode='a', header=header)
