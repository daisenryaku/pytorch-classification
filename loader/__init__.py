from mnist_loader import mnistLoader
from cifar10_loader import cifar10Loader
from imagenet_loader import imagenetLoader


def get_loader(name, data_path, train=True):
    if name == 'mnist':
        return mnistLoader(data_path, train)
    elif name == 'cifar10':
        return cifar10Loader(data_path, train)
    elif name == 'imagenet':
        return imagenetLoader(data_path)
    else:
        raise "Dataset {} not available".format(name)
