"""extract_patches.py

Patch extraction script.
"""

import re
import glob
import os
import tqdm
import pathlib

import numpy as np

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

from dataset import get_dataset

# -------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Determines whether to extract type map (only applicable to datasets with class labels).
    type_classification = True

    win_size = [256, 256]
    step_size = [256, 256]
    extract_type = "mirror"  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.

    # Name of dataset - use Kumar, CPM17 or CoNSeP.
    # This used to get the specific dataset img and ann loading scheme from dataset.py
    
    dataset_name = "monusac"
    save_root = "/data2/zlb/data/MoNuSAC/split231/" 

    # a dictionary to specify where the dataset path should be
    dataset_info = {
        "train": {
            "img": (".tif", "/data2/zlb/data/MoNuSAC/train/unite_tif/"),
            "ann": (".mat", "/data2/zlb/data/MoNuSAC/train/instmat/"),
        },

        "valid": {
            "img": (".tif", "/data2/zlb/data/MoNuSAC/test/unite_tif/"),
            "ann": (".mat", "/data2/zlb/data/MoNuSAC/test/instmat/"),
        },
    }



    patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
    parser = get_dataset(dataset_name)
    xtractor = PatchExtractor(win_size, step_size)
    for split_name, split_desc in dataset_info.items():
        img_ext, img_dir = split_desc["img"]   
        #print(img_ext, img_dir)          #.tif   /data2/zlb/data/MoNuSAC/test/unite_tif/
        ann_ext, ann_dir = split_desc["ann"]
        #print(ann_ext, ann_dir)          #.mat   /data2/zlb/data/MoNuSAC/test/instmat/
        
        out_dir = "%s/%s/%s/%dx%d_%dx%d/" % (
            save_root,
            dataset_name,
            split_name,
            win_size[0],
            win_size[1],
            step_size[0],
            step_size[1],
        )
        file_list = glob.glob(patterning("%s/*%s" % (ann_dir, ann_ext)))
        file_list.sort()  # ensure same ordering across platform
        #print(file_list)
        rm_n_mkdir(out_dir)

        pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbarx = tqdm.tqdm(
            total=len(file_list), bar_format=pbar_format, ascii=True, position=0
        )

        for file_idx, file_path in enumerate(file_list):
            base_name = pathlib.Path(file_path).stem

            img = parser.load_img("%s/%s%s" % (img_dir, base_name, img_ext))
            ann = parser.load_ann(
                "%s/%s%s" % (ann_dir, base_name, ann_ext), type_classification
            )

            # *
            img = np.concatenate([img, ann], axis=-1)
            if img.shape[0]<256 or img.shape[1]<256:
                continue
            sub_patches = xtractor.extract(img, extract_type)

            pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            pbar = tqdm.tqdm(
                total=len(sub_patches),
                leave=False,
                bar_format=pbar_format,
                ascii=True,
                position=1,
            )

            for idx, patch in enumerate(sub_patches):
                np.save("{0}/{1}_{2:03d}.npy".format(out_dir, base_name, idx), patch)
                pbar.update()
            pbar.close()
            # *

            pbarx.update()
        pbarx.close()
