# pytorch-classification

Image Classification architectures implemented in PyTorch. Support MNIST, CIFAR10, ImageNet dataset.

Support [LeNet](#lenet-paper).

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
<a id= 'lenet-paper'>
[1]LeCun Y, Boser B E, Denker J S, et al. Handwritten digit recognition with a back-propagation network[C]//Advances in neural information processing systems. 1990: 396-404.
<br>
