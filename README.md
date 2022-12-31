# Who Should I Engage with At What Time? A Missing Event Aware Temporal Graph Neural Network

This repo contains code for Who Should I Engage with At What Time? A Missing Event Aware Temporal Graph Neural Network (submitted to TNNLS) by Mingyi Liu, Zhiying Tu, Xiaofei Xu and Zhongjie Wang.

## Environment Setup

We exported our experimental environment as `pytorch.yaml`, which you can set up with the following command:

```bash
conda env create -f pytorch.yaml
```

## Run experiments

```bash
python run.py experiment=ENRON.yaml  # pleas find experiment setting in configs/experiments
```

## Notes:
1. The current version is not clean code. 
2. This repo is forked from https://github.com/ashleve/lightning-hydra-template