from torchvision import datasets

trainset = datasets.CIFAR10(root='./data', train=True, download=True)
testset = datasets.CIFAR10(root='./data', train=False, download=True)

# Print dataset information
print(f"Number of classes: {len(trainset.classes)}")
print(f"Training samples: {len(trainset)}")
print(f"Test samples: {len(testset)}")
print(f"Classes: {trainset.classes}")
print(f"Training per class distribution:")
class_counts = {cls: 0 for cls in trainset.classes}
for _, label in trainset:
    class_counts[trainset.classes[label]] += 1
for cls, count in class_counts.items():
    print(f"  {cls}: {count}")

print(f"Training data shape: {trainset.data.shape}")
print(f"Test data shape: {testset.data.shape}")

# Show first image
import matplotlib.pyplot as plt

plt.imshow(trainset.data[0])
plt.title(trainset.classes[trainset.targets[0]])
plt.show()