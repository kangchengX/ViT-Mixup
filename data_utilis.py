import numpy as np
from typing import Tuple

def spilt_data(
    data: Tuple[np.ndarray], 
    ratio_dev: float | None = 0.8, 
    ratio_train: float | None = 0.9
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Split data to training set, validation set and hold-out test set

    Args:
        data (tuple): (images, labels)
        ratio_dev (float): ratio for development set (i.e. training and validation) in the whole data set
        ratio_dev (float): ratio for train set in the whole development set

    Returns:
        ((images_train,labels_train), (images_val,labels_val), (images_test,labels_test))
    """

    # split dataset to development and hold out set
    images, labels = data
    images = images.astype(float) / 255.0
    size = images.shape[0]
    size_dev = int(size * ratio_dev)
    index_new1 = np.random.choice(np.arange(size),size,replace=False)
    images_dev = images[index_new1[0:size_dev]]
    labels_dev = labels[index_new1[0:size_dev]]
    images_test = images[index_new1[size_dev:]]
    labels_test = labels[index_new1[size_dev:]]

    # split development set to train and test set
    size_train = int(size_dev * ratio_train)
    index_new2 = np.random.choice(np.arange(size_dev),size_dev,replace=False)
    images_train = images_dev[index_new2[0:size_train]]
    labels_train = labels_dev[index_new2[0:size_train]]
    images_val = images_dev[index_new2[size_train:]]
    labels_val = labels_dev[index_new2[size_train:]]

    return (images_train,labels_train), (images_val,labels_val), (images_test,labels_test)