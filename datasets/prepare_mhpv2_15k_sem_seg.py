import os
from pathlib import Path

import tqdm
import cv2
from PIL import Image


src_dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "mhpv2_15k"
for name in ["train", "val"]:
    all_files = os.listdir(src_dataset_dir / "images" / name)
    for file in tqdm.tqdm(all_files):
        img = Image.open(src_dataset_dir / "images" / name / file)
        try:
            img = img.convert("RGB")
        except OSError:
            fixed_img = cv2.imread(str(src_dataset_dir / "images" / name / file))
            cv2.imwrite(str(src_dataset_dir / "images" / name / file), fixed_img)
            print("{} is truncated.".format(src_dataset_dir / "images" / name / file))
