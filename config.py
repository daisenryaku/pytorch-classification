#coding:utf8
import warnings
from torchvision import transforms

class DefaultConfig(object):
    dataset_name = 'mnist'
    data_path = '/home/z/classification_data/mnist/'
    img_rows = 28
    num_classes = 10
    model_name = 'LeNet'
    model_save_path = 'checkpoints/LeNet_1106_22:05.ckpt'
    
    batch_size = 128
    test_batch_size = 1000
    lr = 1e-2
    max_epochs = 10
    momentum = 0.5
    save_freq = 10

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
