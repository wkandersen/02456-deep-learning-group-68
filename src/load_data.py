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

    def get_formatted_data(self):
        """
        Get data formatted for neural network training:
        - Images flattened to 1D arrays (32*32*3 = 3072)
        - Pixel values normalized to [0, 1]
        - Labels as numpy arrays
        """
        # Get raw data
        train_data, train_targets = self.get_train_data()
        val_data, val_targets = self.get_validation_data()
        test_data, test_targets = self.get_test_data()
        
        # Flatten images: (N, 32, 32, 3) -> (N, 3072)
        train_data_flat = train_data.reshape(train_data.shape[0], -1)
        val_data_flat = val_data.reshape(val_data.shape[0], -1)
        test_data_flat = test_data.reshape(test_data.shape[0], -1)
        
        # Normalize pixel values to [0, 1]
        train_data_norm = train_data_flat.astype(np.float32) / 255.0
        val_data_norm = val_data_flat.astype(np.float32) / 255.0
        test_data_norm = test_data_flat.astype(np.float32) / 255.0
        
        return (train_data_norm, train_targets), (val_data_norm, val_targets), (test_data_norm, test_targets)

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
    
    # Get formatted data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.get_formatted_data()

    # Get class names
    class_names = data_loader.get_class_names()
    print(f"Class names: {class_names}")
    
    print(f"Unique classes in dataset: {len(np.unique(y_train))}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Validation samples: {len(X_val)}")

    # Print class information
    data_loader.print_class_info()

    print(f"Training per class distribution:")
    class_counts = {cls: 0 for cls in np.unique(y_train)}
    for label in y_train:
        class_counts[label] += 1
    for cls, count in class_counts.items():
        class_name = class_names[cls] if cls < len(class_names) else f"Unknown_{cls}"
        print(f"  {cls} ({class_name}): {count}")

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Validation data shape: {X_val.shape}")

    # Show first image with class name (need to reshape back to 32x32x3 for display)
    import matplotlib.pyplot as plt

    # Reshape flattened image back to 32x32x3 for display
    first_image = X_train[0].reshape(32, 32, 3)
    # Denormalize for proper display (since we normalized to [0,1])
    first_image_display = (first_image * 255).astype(np.uint8)
    
    plt.imshow(first_image_display)
    first_label = y_train[0]
    class_name = class_names[first_label]
    plt.title(f"Class {first_label}: {class_name}")
    plt.show()