# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FreeSOLO/blob/main/LICENSE

#export GLOO_SOCKET_IFNAME=eth0
python train_net.py \
	--dist-url tcp://127.0.0.1:$(( RANDOM % 100 + RANDOM % 10 + 40000 )) \
	--num-gpus 8 \
	--config configs/coco_stuff_lt_40k/semantic-segmentation/maskformer2_R50_bs16_320k.yaml \
	--eval-only MODEL.WEIGHTS coco_stuff_lt_40k_r50/model_final.pth

