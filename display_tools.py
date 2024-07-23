import tensorflow as tf
import numpy as np
from PIL import Image
from models import MixUp
from typing import List, Tuple, Literal

def visualise_mixup(
        images: np.ndarray, 
        size:  Tuple[int], 
        sampling_method: Literal['beta','uniform'], 
        filename: str | None=None, 
        **kwargs
):
    """
    Visualise the mixup results with a montage
    
    Args:
        images (ndarray): image examples
        size (int): size of the montage, (rows, cols)
        sampling_method (str): method to generate lambda. 'beta' indicates beta, 'uniform' indicate uniform
        filename (str | None): path to save the montage. don't save if None.
        alpha (float): parameter for beta distribution
        uniform_range (tuple): predefined range to generate lambda uniformly

    Returns:
        images (ndarray): the montage with the size
    """

    assert images.shape[0] == size[0]*size[1]

    images = tf.cast(images,dtype=np.float32)

    mixup_layer = MixUp(sampling_method,**kwargs)
    images = mixup_layer(images,training=True)

    # generate and save montage
    images = tf.concat([tf.concat([images[i+4*j,...] for i in range(size[1])],1) for j in range(size[0])],0)
    images = Image.fromarray(images.numpy().astype(np.uint8))

    if filename is not None:
        images.save(filename)

    return images


def visualise_results(
        filename_model: str, 
        filename_fig: str, 
        images: np.ndarray, 
        labels: np.ndarray, 
        class_names: List[str]):
    """
    Visualise results of the trained model by saving the montage and printing the ture and predicted labels
    
    Args: 
        filename_model: path of the saved model
        filename_fig: path to save the montage
        images: a tensor with shape (num, height, width, chanels)
        labels: labels of the images
        class_name: true names of the classes
    
    """

    num = images.shape[0]
    montage_width = int(np.sqrt(num))
    assert num % montage_width == 0, 'num of images should be a  quare number'
    assert images.shape[0] == labels.shape[0]

    images_norm = images.astype(float) / 255.0

    print('\n----------start : load model and predict------------')
    model = tf.keras.models.load_model(filename_model)
    preds = np.argmax(model.predict(images_norm),1)

    print('Ground-truth:' + ' '.join('%5s' % class_names[labels[j,0]] for j in range(num)))
    print('Predicted: ', ' '.join('%5s' % class_names[preds[j]] for j in range(num)))
    print('----------finish : load model and predict------------\n')

    # generate and save montage
    images = np.concatenate([np.concatenate([images[i+montage_width*j,...] 
                                             for i in range(montage_width)],1) 
                             for j in range(montage_width)],0)
    images = Image.fromarray(images)
    images.save(filename_fig)


def report_summary(
        record: Tuple[np.ndarray] | float, 
        dataset_type: Literal['train', 'validation', 'test']
):
    """
    Report summary of results of the model

    Args: 
        record (tuple): (loss_array, accuracy_array, time_array) for train and validation, 
            accuracy_scaler for hold out test
        dataset_type (str): train, validation or test
    """

    if dataset_type == 'test':
       # test
       print('''\ton hold out test set: accuracy: {:.3f}%'''.format(record*100))
    
    elif dataset_type in ['train','validation']:
        # development
        if dataset_type == 'train':
            accuracy_name = 'aug_accuracy'
        else:
            accuracy_name = 'accuracy'
        print('''\ton {} set:
            lowest loss: {:.3f} at epoch {}
            final loss: {:.3f}
            higest {}: {:.3f}% at epoch {}
            final {}: {:.3f}%
            average time: {:.3f} s/epoch
        '''.format(dataset_type,
                np.min(record[0]), 1+np.argmin(record[0]),
                record[0][-1],
                accuracy_name, np.max(record[1])*100, 1+np.argmax(record[1]),
                accuracy_name, record[1][-1]*100,
                np.mean(record[2])))
        
    else:
        raise ValueError(f'unknown dataset type : {dataset_type}')
