# Learning Deformable Image Registration from Optimization: Perspective, Modules, Bilevel Training and Beyond

This is the official Pytorch implementation of "Learning Deformable Image Registration from Optimization: Perspective, Modules, Bilevel Training and Beyond" (IEEE TPAMI 2021).

<!-- ![Alt text](pipeline.png) -->


## Prerequisites
- `Python 3.6.8+`
- `Pytorch 0.3.1`
- `torchvision 0.2.0`
- `NumPy`
- `NiBabel`

This code has been tested with `Pytorch` and GTX1080TI GPU.


## Inference
```
python MultiModal/test.py 
```
A pretrained MultiPropReg model is available in "models/MPR-T1-to-T2atlas.zip".

## Training
If you want to train a new MultiPropReg model using your own dataset, please define your own data generator for `train_t1atlas.py` and perform the following script.
```
python MultiModal/train_t1atlas.py
```

## Publication
If you find this repository useful, please cite:

- **Learning Deformable Image Registration from Optimization: Perspective, Modules, Bilevel Training and Beyond**  
R. Liu, [Z. Li](https://alison-brie.github.io/), X. Fan, C. Zhao, H. Huang and Z. Luo. IEEE TPAMI [eprint arXiv:2004.14557](https://arxiv.org/abs/2004.14557)

## Acknowledgment
Some codes in this repository are modified from [VoxelMorph](https://github.com/voxelmorph/voxelmorph).

## Keywords
Diffeomorphic Image Registration, hyperparameter Learning in Registration, convolutional neural networks, alignment
