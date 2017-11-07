from LeNet import LeNet
from AlexNet import AlexNet
from VGG import VGG16

def get_model(name, in_channels, img_rows, num_classes):
    model = _get_model_instance(name)
    if name == 'LeNet':
        model = model(in_channels=in_channels, img_rows=img_rows, num_classes=num_classes)
    elif name == 'AlexNet':
        model = model(in_channels=in_channels, img_rows=img_rows, num_classes=num_classes)
    elif name == 'VGG':
        model = model(in_channels=in_channels, img_rows=img_rows, num_classes=num_classes)
    else:
        raise 'Model {} not available'.format(name)
    return model

def _get_model_instance(name):
    return {
            'LeNet':LeNet,
            'AlexNet':AlexNet,
            'VGG':VGG16,
        }[name]
