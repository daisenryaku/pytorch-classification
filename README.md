# pytorch-classification

Image Classification architectures implemented in PyTorch. Support MNIST, CIFAR10, ImageNet dataset.

Support [LeNet](#lenet-paper), [AlexNet](#alexnet-paper), [VGG](#vgg-paper), [ResNet](#resnet-paper), [DenseNet](#densenet-paper).

## Requirements
* torch
* torchvision

**Note**:

You can install all the python packages you needed by running:
```bash
sudo pip install -r requirements.txt
```

## Setup data
For MNIST and CIFAR10 dataset: open **config.py**, change the **dataset_name**, **data_path**, **model_name** .

For ImageNet dataset: download the ImageNet dataset and move validation images to labeled subfolders. See [this](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset)

## Train model
```bash
python main.py train
```

## Test
```bash
python main.py test
```

## Citations
<a id = 'lenet-paper'>
[1]LeCun Y, Boser B E, Denker J S, et al. Handwritten digit recognition with a back-propagation network[C]//Advances in neural information processing systems. 1990: 396-404.
<br>

<a id = 'alexnet-paper'>
[2]Krizhevsky A, Sutskever I, Hinton G E. Imagenet classification with deep convolutional neural networks[C]//Advances in neural information processing systems. 2012: 1097-1105.
<br>

<a id = 'vgg-paper'>
[3]Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition[J]. arXiv preprint arXiv:1409.1556, 2014.
<br>

<a id = 'resnet-paper'>
[4]He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.
<br>

<a id = 'densenet-paper'>
[5]Huang G, Liu Z, Weinberger K Q, et al. Densely connected convolutional networks[J]. arXiv preprint arXiv:1608.06993, 2016.
<br>
