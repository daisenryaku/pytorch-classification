#coding:utf8
import warnings
from torchvision import transforms

class DefaultConfig(object):
    dataset_name = 'imagenet'
    data_path = '/home/z/classification_data/imagenet/'
    in_channels = 3
    img_rows = 224
    num_classes = 2
    model_name = 'AlexNet'
    model_save_path = 'checkpoints/AlexNet_1107_09:28.ckpt'
    
    batch_size = 32
    test_batch_size = 32
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
