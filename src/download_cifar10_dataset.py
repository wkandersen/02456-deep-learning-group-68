from torchvision import datasets

trainset = datasets.CIFAR10(root='./data', train=True, download=True)
testset = datasets.CIFAR10(root='./data', train=False, download=True)

# Print dataset information
print(f"Number of classes: {len(trainset.classes)}")
print(f"Training samples: {len(trainset)}")
print(f"Test samples: {len(testset)}")