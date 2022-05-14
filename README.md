# Simple Image Classification

## Introduction

This program performs IMAGE CLASSIFICATION on a simple deep learning pipeline.
This pipeline is set to ImageNet-1K and this can be changed to another one

## Model

- AlexNet
- VGGNet(11,13,16,19)
- ResNet50
- EfficientNet(b0,b1,b2,b3,b4,b5,b6,b7)

## Script

### Train

```
$ python3 run/run_train.py \
        --config-file {yaml file path} \
        --wandb-key {wandb key} \
        --training-keyword {keyword for the run}
```