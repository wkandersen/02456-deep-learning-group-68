import numpy as np
import os
import pickle


class CIFAR10Loader:
    def __init__(self, data_dir='data/cifar-10-batches-py'):
        self.data_dir = data_dir
        self.images = None
        self.labels = None
        self.decoded_labels = None
        self._load_data()


    # Data loader
    def _unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def _load_data(self):
        ###                             ###
        ### Combine dataset batches 1-5 ###
        ###                             ###
        # Initialize lists to hold all data and labels
        all_data = []
        all_labels = []

        # Load and combine all 5 batches
        for i in range(2, 6):
            batch = self._unpickle(f'data/cifar-10-batches-py/data_batch_{i}')
            data = batch[b'data']
            labels = batch[b'labels']
            all_data.append(data)
            all_labels.extend(labels)

        # Convert to numpy arrays
        self.train_images = np.concatenate(all_data)
        self.train_labels = np.array(all_labels)

        # Load class label names
        meta = self._unpickle("data/cifar-10-batches-py/batches.meta")
        self.decoded_labels = [label.decode('utf-8') for label in meta[b'label_names']]

        # print(f"Combined data shape: {self.images.shape}")
        # print(f"Combined labels shape: {self.labels.shape}")

        test = self._unpickle('data/cifar-10-batches-py/test_batch')
        self.test_images = test[b'data']
        self.test_labels = np.array(test[b'labels'])


    def get_validation_data(self, batch_number="data_batch_1"):

        # Load the specified batch
        batch = self._unpickle(f'data/cifar-10-batches-py/{batch_number}')
        self.val_images = batch[b'data']
        self.val_labels = np.array(batch[b'labels'])
        return self.val_images, self.val_labels

    def get_data(self):
        return self.train_images, self.train_labels, self.decoded_labels

    def get_training_data(self):
        return self.train_images, self.train_labels

    def get_test_data(self):
        return self.test_images, self.test_labels

    ###                  ###
    ### Display an image ###
    ###                  ###
    def display(self, img_num: int):
        import matplotlib.pyplot as plt
        # Access the first image and label
        first_image = self.images[img_num]            # a 3072-length array
        first_label = self.labels[img_num]            # an integer label

        # Reshape the image: 3072 = 1024 R + 1024 G + 1024 B
        r = first_image[0:1024].reshape(32, 32)
        g = first_image[1024:2048].reshape(32, 32)
        b = first_image[2048:].reshape(32, 32)

        # Stack channels to form RGB image
        img = np.stack([r, g, b], axis=2)

        plt.imshow(img)
        plt.title(f"Label: {self.decoded_labels[first_label]}")
        plt.show()



## How to use the class.

# loader = CIFAR10Loader()
# images, labels, decoded_labels = loader.get_data()

# print(images.shape)         # (50000, 3072)
# print(labels.shape)         # (50000,)
# print(decoded_labels)       # ['airplane', 'automobile', ..., 'truck']

# loader.display(1)           # Show image 0



dataloader = CIFAR10Loader()
train_images, train_labels = dataloader.get_training_data()
test_images, test_labels = dataloader.get_test_data()
validation_images, validation_labels = dataloader.get_validation_data()



print(f"Training data shape: {train_images.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Validation data shape: {validation_images.shape}")
print(f"Validation labels shape: {validation_labels.shape}")
print(f"Test data shape: {test_images.shape}")
print(f"Test labels shape: {test_labels.shape}")
