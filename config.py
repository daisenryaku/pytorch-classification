#coding:utf8
import warnings
from torchvision import transforms

class DefaultConfig(object):
    dataset_name = 'cifar10'
    data_path = '/home/z/classification_data/cifar10/'
    in_channels = 3
    img_rows = 32
    num_classes = 10
    model_name = 'VGG'
    model_save_path = 'checkpoints/AlexNet_1107_09:28.ckpt'
    
    batch_size = 1
    test_batch_size = 1
    lr = 1e-2
    momentum = 0.5
    max_epochs = 5
    save_freq = 5

    def parse(self, kwargs):
        for k, v in kwargs.iteritems():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute {}".format(k))
                setattr(self, k, v)
        print('user config:')
        for k, v in self.__class__.__dict__.iteritems():
            if not k.startswith('__'):
                print (k, getattr(self, k))

opt = DefaultConfig()
