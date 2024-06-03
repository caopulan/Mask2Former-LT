import numpy as np

import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')
from ltss.data.datasets import MHPv2_CATEGORIES
from ltss.data.datasets import COCO_CATEGORIES
from ltss.data.datasets import ADE20K_SEM_SEG_FULL_CATEGORIES


meta_files = {
    "mhpv2-lt": MHPv2_CATEGORIES,
    "ade20k-full": ADE20K_SEM_SEG_FULL_CATEGORIES,
    "coco-lt": COCO_CATEGORIES
}

for dataset_name, meta_file in meta_files.items():
    image_results = {'frequent':[], 'common': [], 'rare':[]}
    pixel_results = {'frequent':[], 'common': [], 'rare':[]}
    for category in meta_file:
        id = category['trainId']
        image_freq = category['image_frequency']
        pixel_freq = category['pixel_frequency']
        image_results[image_freq].append(id)
        pixel_results[pixel_freq].append(id)

    for key, value in image_results.items():
        np.save(f'ltss/data/{dataset_name}_pixel-{key}_classid.npy', np.array(value))
        np.save(f'ltss/data/{dataset_name}_image-{key}_classid.npy', np.array(value))
    print()