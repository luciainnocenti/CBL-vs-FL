import itk
import SimpleITK as sitk
import os
import shutil
from pathlib import Path
import sys
HOME = str(Path.home())
rootdir = sys.argv[1]

for dir, subdirs, files in os.walk(rootdir):
    print(f"dir = {dir}, subdir =  {subdirs}")
    if len(files) != 0 and str.split(dir, "/")[-1] in ["image", "label"]:
        i = itk.imread(os.path.join(dir, files[0]))
        np_view = itk.array_view_from_image(i)
        if str.split(dir, "/")[-1] == "image":
            np_view = np_view[0]
        new_img = sitk.GetImageFromArray(np_view)
        orientation = sitk.FlipImageFilter()
        orientation.SetFlipAxes([False, True, False])
        flipped_img = orientation.Execute(new_img)

        newdir = str.replace(dir, "decathlon_original/dataset", "datasets_pp/decathlon")
        os.makedirs(newdir)
        sitk.WriteImage(flipped_img, os.path.join(newdir, files[0]))

participants = f"{rootdir}/participants.csv"
shutil.copy(participants, str.replace(participants, "decathlon_original/dataset", "datasets_pp/decathlon"))
