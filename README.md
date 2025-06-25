# KD-DETR
This repository is the official implementation of the CVPR 2024 paper KD-DETR: Knowledge Distillation for Detection Transformer with Consistent Distillation Points Sampling.
[![arXiv](https://img.shields.io/badge/Arxiv-2407.11335-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2211.08071)
![](assets/17508352785072.png)
## Installation
```
python==3.9.12
torch==1.9.1+cu111
torchvision==0.10.1+cu111
```
## Training
### Prepare
Download the [MS-COCO](https://cocodataset.org/#download) to data/coco/
Download the [DAB-DETR](https://github.com/IDEA-Research/DAB-DETR?tab=readme-ov-file) teacher models to pretrained/
```
KD-DETR/
|-data/
|--coco/
|--custom_data/
|-pretrained/
|--dab_detr_r50.pth
|--custom_teacher_model
```
### Scripts
Distillation training with DAB-DETR-R18 as student model and DAB-DETR-R50 as teacher model:
```
sh tools/train.sh
```

