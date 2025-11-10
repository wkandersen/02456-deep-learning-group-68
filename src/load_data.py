from torchvision import datasets
import numpy as np

class DataLoader:
    def __init__(self, root='./data', validation_split=0.1):
        self.trainset = datasets.CIFAR10(root=root, train=True, download=True)
        self.testset = datasets.CIFAR10(root=root, train=False, download=True)
        self.validation_split = validation_split
        self.val_size = int(len(self.trainset) * self.validation_split)


    def get_validation_data(self):
        # Using part of the training set as validation set
        self.val_size = int(len(self.trainset) * self.validation_split)
        val_data = np.array(self.trainset.data[:self.val_size])
        val_targets = np.array(self.trainset.targets[:self.val_size])
        return val_data, val_targets

    def get_train_data(self):
        train_data = np.array(self.trainset.data[self.val_size:])
        train_targets = np.array(self.trainset.targets[self.val_size:])
        return train_data, train_targets
    
    def get_test_data(self):
        return np.array(self.testset.data), np.array(self.testset.targets)
    
    
    def get_class_names(self):
        """Get the list of class names"""
        return self.trainset.classes
    
    def get_num_classes(self):
        """Get the number of classes"""
        return len(self.trainset.classes)
    
    def print_class_info(self):
        """Print information about classes"""
        class_names = self.get_class_names()
        print(f"Number of classes: {len(class_names)}")
        print("Class names:")
        for i, class_name in enumerate(class_names):
            print(f"  {i}: {class_name}")
        return class_names



if __name__ == "__main__":
    data_loader = DataLoader()
    train_images, train_labels = data_loader.get_train_data()
    test_images, test_labels = data_loader.get_test_data()
    val_images, val_labels = data_loader.get_validation_data()
    
    # Get class names
    class_names = data_loader.get_class_names()
    print(f"Class names: {class_names}")
    
    print(f"Unique classes in dataset: {len(np.unique(train_labels))}")
    print(f"Training samples: {len(train_images)}")
    print(f"Test samples: {len(test_images)}")
    print(f"Validation samples: {len(val_images)}")

    # Print class information
    data_loader.print_class_info()

    print(f"Training per class distribution:")
    class_counts = {cls: 0 for cls in np.unique(train_labels)}
    for label in train_labels:
        class_counts[label] += 1
    for cls, count in class_counts.items():
        class_name = class_names[cls] if cls < len(class_names) else f"Unknown_{cls}"
        print(f"  {cls} ({class_name}): {count}")

    print(f"Training data shape: {train_images.shape}")
    print(f"Test data shape: {test_images.shape}")
    print(f"Validation data shape: {val_images.shape}")

    # Show first image with class name
    import matplotlib.pyplot as plt

    plt.imshow(train_images[0])
    first_label = train_labels[0]
    class_name = class_names[first_label]
    plt.title(f"Class {first_label}: {class_name}")
    plt.show()