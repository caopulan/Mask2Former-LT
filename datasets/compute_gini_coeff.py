import argparse
import os
import cv2
import numpy as np
import tqdm
from PIL import Image
from pathlib import Path


def get_args_parser():
    parser = argparse.ArgumentParser('LTSS', add_help=False)

    parser.add_argument('--dataset', default='ade20k', type=str,
                        choices=['ade20k', 'ade20k_full', 'coco_stuff', 'coco_stuff_lt', 'mhpv2', 'mhpv2_lt'])
    parser.add_argument('--set', default='train', type=str,
                        choices=['train', 'val'])
    return parser


def compute_gini(x, w=None):
    # The rest of the code requires numpy arrays.
    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) /
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def get_dataset_meta(args):
    dataset_root = Path(os.getenv("DETECTRON2_DATASETS", "datasets"))
    if args.dataset == "ade20k":
        anno_dir = dataset_root / "ADEChallengeData2016/annotations_detectron2"
        return anno_dir / "training" if args.set == "train" else anno_dir / "validation", 150
    if args.dataset == "ade20k_full":
        anno_dir = dataset_root / "ADE20K_2021_17_01/annotations_detectron2"
        return anno_dir / "training" if args.set == "train" else anno_dir / "validation", 847
    if args.dataset == "coco_stuff":
        anno_dir = dataset_root / "coco_stuff_118k/annotations"
        return anno_dir / "train2017" if args.set == "train" else anno_dir / "val2017", 171
    if args.dataset == "coco_stuff_lt":
        anno_dir = dataset_root / "coco_stuff_lt_40k/annotations"
        return anno_dir / "train2017" if args.set == "train" else anno_dir / "val2017", 171
    if args.dataset == "mhpv2":
        anno_dir = dataset_root / "mhpv2_15k/annotations"
        return anno_dir / "train" if args.set == "train" else anno_dir / "val", 59
    if args.dataset == "mhpv2_lt":
        anno_dir = dataset_root / "mhpv2_lt_7k/annotations"
        return anno_dir / "train" if args.set == "train" else anno_dir / "val", 59


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Compute gini coefficient', parents=[get_args_parser()])
    args = parser.parse_args()

    annotation_dir, num_classes = get_dataset_meta(args)
    num_images = len(list(annotation_dir.iterdir()))

    image_mat = np.zeros((num_images, num_classes), dtype=np.float32)
    pixel_mat = np.zeros((num_images, num_classes), dtype=np.float32)
    for idx, file in enumerate(tqdm.tqdm(list(annotation_dir.iterdir()))):
        if args.dataset == "ade20k_full":
            png = Image.open(file)
            png = np.asarray(png)
            uni = np.unique(png)
            uni = np.setdiff1d(uni, [65535])
        else:
            png = cv2.imread(str(file), 0)
            uni = np.unique(png)
            uni = np.setdiff1d(uni, [255])

        tmp_image = np.zeros(num_classes)
        tmp_image[uni] = 1
        image_mat[idx] = tmp_image

        tmp_pixel = np.zeros(num_classes)
        all_pixel = png.shape[0] * png.shape[1]
        for j in uni:
            mask = np.where(png.copy() == j, 1, 0)
            pixel_ratio = np.sum(mask) / float(all_pixel)
            tmp_pixel[j] += pixel_ratio
        pixel_mat[idx] = tmp_pixel

    image_sum = np.sum(image_mat, axis=0)
    pixel_sum = np.sum(pixel_mat, axis=0)

    print("******************* Image-level *******************")
    print("max@image", max(image_sum))
    print("min@image", min(image_sum))
    print("gini@image", compute_gini(image_sum))

    print("******************* Pixel-level *******************")
    print("max@pixel", max(pixel_sum))
    print("min@pixel", min(pixel_sum))
    print("gini@pixel", compute_gini(pixel_sum))
