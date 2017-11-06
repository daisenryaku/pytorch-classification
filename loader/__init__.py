from mnist_loader import mnistLoader

def get_loader(name, data_path, train=True):
    return {
            'mnist' : mnistLoader(data_path, train),
            }[name]
