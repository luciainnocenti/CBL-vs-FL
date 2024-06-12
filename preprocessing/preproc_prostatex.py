import itk
import SimpleITK as sitk
import os
import shutil
from pathlib import Path
import sys
HOME = str(Path.home())

rootdir = sys.argv[1]

for dirs, subdirs, files in os.walk(rootdir):
    if len(files) != 0 and str.split(dirs, "/")[-1] in ["image", "label"]:
        newdir = str.replace(dirs, "prostateX_original/dataset", "datasets_pp_nv/prostatex_dataset_pp")
        os.makedirs(newdir)
        print(f'newdir = {newdir}')
        if str.split(dirs, "/")[-1] == "label":
            i = itk.imread(os.path.join(dirs, files[0]), itk.SI)
        else:
            i = itk.imread(os.path.join(dirs, files[0]))
        np_view = itk.array_view_from_image(i)
        new_img = sitk.GetImageFromArray(np_view)
        orientation = sitk.FlipImageFilter()
        orientation.SetFlipAxes([False, True, False])
        flipped_img = orientation.Execute(new_img)

        sitk.WriteImage(flipped_img, os.path.join(newdir, files[0]))

participants = f"{rootdir}/participants.csv"
shutil.copy(participants, str.replace(participants, "prostateX_original/dataset",
                                      "datasets_pp_nv/prostatex"))
