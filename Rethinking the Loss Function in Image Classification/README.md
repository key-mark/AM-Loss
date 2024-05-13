# Rethinking the Loss Function in Image Classification
This is a DEMO using AM loss for image classification. Among them, we provide a code running in the CIFAR10 data set. You can try to use ordinary image classification models, such as ResNet18 and VGG models. Similarly, we also demonstrate the improvement of AM losses for the transform architecture model. You can choose to train a VIT model from the beginning, or to train the VIT standard model of downloading pre -trained from HuggingFace.
Using the repository is straightforward - all you need to do is run the `train_cifar10.py` script with different arguments, depending on the model and training parameters you'd like to use.


# Usage example
`python train_cifar10.py` # vit-patchsize-4

`python train_cifar10.py  --size 48` # vit-patchsize-4-imsize-48

`python train_cifar10.py --patch 2` # vit-patchsize-2

`python train_cifar10.py --net vit_small --n_epochs 400` # vit-small

`python train_cifar10.py --net vit_timm` # train with pretrained vit

`python train_cifar10.py --net convmixer --n_epochs 400` # train with convmixer

`python train_cifar10.py --net mlpmixer --n_epochs 500 --lr 1e-3`

`python train_cifar10.py --net cait --n_epochs 200` # train with cait

`python train_cifar10.py --net swin --n_epochs 400` # train with SwinTransformers

`python train_cifar10.py --net res18` # resnet18+randaug

# Used in..
* Vision Transformer Pruning [arxiv](https://arxiv.org/abs/2104.08500) [github](https://github.com/Cydia2018/ViT-cifar10-pruning)
