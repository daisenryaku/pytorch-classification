from LeNet import LeNet
from AlexNet import AlexNet
from VGG import VGG16
from ResNet import resnet18
from DenseNet import densenet121

def get_model(name, in_channels=3, img_rows=224, num_classes=2):
    model = _get_model_instance(name)
    if name == 'LeNet':
        model = model(in_channels=in_channels, img_rows=img_rows, num_classes=num_classes)
    elif name == 'AlexNet':
        model = model(in_channels=in_channels, img_rows=img_rows, num_classes=num_classes)
    elif name == 'VGG':
        model = model(in_channels=in_channels, img_rows=img_rows, num_classes=num_classes)
    elif name == 'ResNet':
        model = model(in_channels=in_channels, num_classes=num_classes)
    elif name == 'DenseNet':
        model = model()
    else:
        raise 'Model {} not available'.format(name)
    return model

def _get_model_instance(name):
    return {
            'LeNet':LeNet,
            'AlexNet':AlexNet,
            'VGG':VGG16,
            'ResNet':resnet18,
            'DenseNet':densenet121,
        }[name]
