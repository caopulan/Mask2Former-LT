## Getting Started with LTSS

This document provides a brief intro of the usage of LTSS.

Please follow [datasets/README.md](./datasets/README.md) to prepare dataset first.


### Training & Evaluation in Command Line

We provide a script `train_net.py`, that is made to train all the configs provided in LTSS.

You can use corresponding config to train all long-tailed method (**RFS/CopyPaste/SeesawLoss/Frequency-based Matcher**) and our **advanced benchmark**.

For example, to train advanced benchmark on ADE20K-Full, run:
```
python train_net.py --num-gpus 8 \
  --config-file configs/ade20k_full/mask2former/benchmark-maskformer2_R50_bs16_200k.yaml
```

To evaluate a model's performance, use
```
python train_net.py \
  --config-file configs/ade20k_full/mask2former/benchmark-maskformer2_R50_bs16_200k.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```
For more options, see `python train_net.py -h`.
