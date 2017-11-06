# pytorch-classification

Classification on MNIST dataset with Pytorch.

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
Open **config.py**, change the **dataset_name**, **data_path**, **model_name** .

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
