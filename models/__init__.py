from LeNet import LeNet
from AlexNet import AlexNet

def get_model(name, img_rows, num_classes):
    model = _get_model_instance(name)
    if name == 'LeNet':
        model = model(img_rows=img_rows, num_classes=num_classes)
    elif name == 'AlexNet':
        model = model(img_rows=img_rows, num_classes=num_classes)
    else:
        raise 'Model {} not available'.format(name)
    return model

def _get_model_instance(name):
    return {
            'LeNet':LeNet,
            'AlexNet':AlexNet,
        }[name]
