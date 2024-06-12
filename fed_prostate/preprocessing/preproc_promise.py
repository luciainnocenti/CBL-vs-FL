import itk
import os
import SimpleITK as sitk
from pathlib import Path
import shutil
import numpy as np
from tqdm import tqdm
import sys

def get_progress_bar_len(dir_path):
    os.system("find {} -type f | wc -l > ./tmp_shell_output".format(dir_path))
    f = open("./tmp_shell_output")
    n = int(f.readlines()[0])
    f.close()
    os.remove("./tmp_shell_output")
    return n


def apply_preproc(rootdir: str):
    n = get_progress_bar_len(rootdir)
    for dirs, subdirs, files in tqdm(os.walk(rootdir), total=n):
        if len(files) != 0 and str.split(dirs, "/")[-1] in ["image", "label"]:
            newdir = str.replace(dirs, "promise_original/dataset", "datasets_pp/promise_dataset_pp")
            if os.path.exists(newdir):
                print(f"Existing dir = {newdir}")
            else:
                os.makedirs(newdir)
            if str.split(dirs, "/")[-1] == "label":
                i = itk.imread(os.path.join(dirs, files[0]), itk.SI)
                np_view = itk.array_view_from_image(i)
                new_i = itk.image_from_array(np_view.transpose())
                itk.imwrite(new_i, os.path.join(newdir, files[0]))
            else:
                i = sitk.ReadImage(os.path.join(dirs, files[0]))  # itk.imread(os.path.join(dirs, files[0]))
                mask_image = sitk.OtsuThreshold(i, 0, 1, 200)
                corrector = sitk.N4BiasFieldCorrectionImageFilter()
                corrected_image = corrector.Execute(i, mask_image)

                iwif = sitk.IntensityWindowingImageFilter()
                array_ci = sitk.GetArrayFromImage(corrected_image)
                q_max = np.quantile(array_ci, 0.975)
                q_min = np.quantile(array_ci, 0.025)
                iwif.SetWindowMaximum(q_max)
                iwif.SetWindowMinimum(q_min)
                iwif.SetOutputMinimum(0)
                iwif.SetOutputMaximum(1)
                iwifed_image = iwif.Execute(corrected_image)
                sitk.WriteImage(iwifed_image, os.path.join(newdir, files[0]))

    return


HOME = str(Path.home())
file_directory = sys.argv[1]
shutil.rmtree(str.replace(file_directory, "promise_original/dataset", "datasets_pp_v2/promise_dataset_pp"), ignore_errors=True)

apply_preproc(file_directory)
participants = f"{file_directory}/participants.csv"
shutil.copy(participants, str.replace(participants, "promise_original/dataset", "datasets_pp/promise_dataset_pp"))
