from torchvision import datasets, transforms

transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))]
        )

def mnistLoader(data_path, train):
    return datasets.MNIST(root=data_path, train=train, download=True, transform=transform)
