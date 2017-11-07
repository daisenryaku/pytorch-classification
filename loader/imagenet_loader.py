from torchvision import datasets, transforms

normalize = transforms.Normalize(
		mean=[0.485, 0.456, 0.406],
		std= [0.229, 0.224, 0.225])

transform = transforms.Compose([
		transforms.RandomSizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalize,
		])

def imagenetLoader(data_path):
	return datasets.ImageFolder(root=data_path,  transform=transform)
	
