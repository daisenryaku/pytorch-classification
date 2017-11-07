from torchvision import datasets, transforms

transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


def cifar10Loader(data_path, train):
    return datasets.CIFAR10(root=data_path, train=train, download=True, transform=transform)
