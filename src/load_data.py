from torchvision import datasets
import numpy as np
from keras.datasets import fashion_mnist

class DataLoaderCifar10:
    def __init__(self, root='./data', validation_split=0.1):
        self.trainset = datasets.CIFAR10(root=root, train=True, download=True)
        self.testset = datasets.CIFAR10(root=root, train=False, download=True)
        self.validation_split = validation_split
        self.val_size = int(len(self.trainset) * self.validation_split)


    def _get_validation_data(self):
        # Using part of the training set as validation set
        self.val_size = int(len(self.trainset) * self.validation_split)
        val_data = np.array(self.trainset.data[:self.val_size])
        val_targets = np.array(self.trainset.targets[:self.val_size])
        return val_data, val_targets

    def _get_train_data(self):
        train_data = np.array(self.trainset.data[self.val_size:])
        train_targets = np.array(self.trainset.targets[self.val_size:])
        return train_data, train_targets
    
    def _get_test_data(self):
        return np.array(self.testset.data), np.array(self.testset.targets)

    def get_formatted_data(self):
        """
        Get data formatted for neural network training:
        - Images flattened to 1D arrays (32*32*3 = 3072)
        - Pixel values normalized to [0, 1]
        - Labels as numpy arrays
        """
        # Get raw data
        train_data, train_targets = self._get_train_data()
        val_data, val_targets = self._get_validation_data()
        test_data, test_targets = self._get_test_data()

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

    def create_subset(self, split_ratio=0.25):
        """Create a subset of the training data for quick experiments"""
        (train_data, train_targets), _, _ = self.get_formatted_data()
        subset_size = int(len(train_data) * split_ratio)
        subset_data = train_data[:subset_size]
        subset_targets = train_targets[:subset_size]

        return subset_data, subset_targets



class DataLoaderFashionMNIST:
    def __init__(self, validation_split=0.1):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = fashion_mnist.load_data()
        self.validation_split = validation_split
        self.val_size = int(len(self.X_train) * self.validation_split)

    def _get_validation_data(self):
        val_data = self.X_train[:self.val_size]
        val_targets = self.y_train[:self.val_size]
        return val_data, val_targets

    def _get_train_data(self):
        train_data = self.X_train[self.val_size:]
        train_targets = self.y_train[self.val_size:]
        return train_data, train_targets
    
    def _get_test_data(self):
        return self.X_test, self.y_test
    
    def get_data(self):
        x_train, y_train = self._get_train_data()
        x_val, y_val = self._get_validation_data()
        x_test, y_test = self._get_test_data()
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
        
    def get_class_names(self):
        """Get the list of class names"""
        return [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
    
    def print_class_info(self):
        """Print information about classes"""
        class_names = self.get_class_names()
        print(f"Number of classes: {len(class_names)}")
        print("Class names:")
        for i, class_name in enumerate(class_names):
            print(f"  {i}: {class_name}")
        return class_names
    
    def get_num_classes(self,):
        """Get the number of classes"""
        return 10
    
    def create_subset(self, split_ratio=0.25):
        """Create a subset of the training data for quick experiments"""
        train_data, train_targets = self._get_train_data()
        subset_size = int(len(train_data) * split_ratio)
        subset_data = train_data[:subset_size]
        subset_targets = train_targets[:subset_size]

        return subset_data, subset_targets





if __name__ == "__main__":


    dataset_choice = input("Write cifar or fashion to choose dataset loader: \n")
    if dataset_choice.lower() == "cifar":
        dataload_cifar = DataLoaderCifar10()
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataload_cifar.get_formatted_data()

    elif dataset_choice.lower() == "fashion":
        dataload_fashion = DataLoaderFashionMNIST()
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataload_fashion.get_data()
    else:
        print("Invalid choice, defaulting to CIFAR-10 loader.")
        dataload_cifar = DataLoaderCifar10()
    
    subset_or_not = input("Subset?: Yes or no \n")
    # Get formatted data
    if subset_or_not == "yes":
        x_subset, y_subset = dataload_cifar.create_subset(split_ratio=0.25)
        print(f"Subset data shape: {x_subset.shape}, Subset labels shape: {y_subset.shape}")


    if dataset_choice.lower() == "cifar":
    # Get class names
        class_names = dataload_cifar.get_class_names()

    if dataset_choice.lower() == "fashion":
        class_names = dataload_fashion.get_class_names()
    print(f"Class names: {class_names}")
    
    print(f"Unique classes in dataset: {len(np.unique(y_train))}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Validation samples: {len(X_val)}")

    # Print class information
    if dataset_choice.lower() == "cifar":
        dataload_cifar.print_class_info()

    if dataset_choice.lower() == "fashion":
        dataload_fashion.print_class_info()

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
    if dataset_choice.lower() == "cifar":
        # first_image = X_train[0].reshape(32, 32, 3)
        # # Denormalize for proper display (since we normalized to [0,1])
        # first_image_display = (first_image * 255).astype(np.uint8)

        # show 10 unique classes in subplot after denormalizing
        for unique_class in range(10):
            # Find the first occurrence of the unique class
            for i, label in enumerate(y_train):
                if label == unique_class:
                    break
            plt.subplot(2, 5, unique_class+1)
            # Denormalize for proper display (since we normalized to [0,1])
            denormalized_image = (X_train[unique_class] * 255).astype(np.uint8)
            plt.imshow(denormalized_image.reshape(32, 32, 3))
            first_label = y_train[unique_class]
            class_name = class_names[first_label]
            plt.title(f"Class {first_label}: {class_name}")
            plt.axis('off')  # Hide axes for cleaner display
        plt.show()

    if dataset_choice.lower() == "fashion":
        plt.imshow(X_train[0], cmap='gray')
        first_label = y_train[0]
        class_name = class_names[first_label]
        plt.title(f"Class {first_label}: {class_name}")
        plt.show()
