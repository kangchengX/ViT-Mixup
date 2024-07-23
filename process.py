import tensorflow as tf
import time, json, os
import numpy as np
from tensorflow.keras.optimizers import Optimizer
from models import VitAug
from typing import Tuple
from warnings import warn


def index_batch_generate(size: int, batch_size: int):
    """
    Generate index for each batch withn an epoch
    
    Args:
        size (int): size of the data wihin one epoch
        batch_size (int): size of the batch
    
    Returns:
        index_batch (list): a list of indices
    """
    assert size > batch_size, f'batch {batch_size} is bigger than data size {size}'

    num_iter = size // batch_size
    batch_left = size % batch_size
    index_epoch = np.random.choice(size,size,replace=False)
    index_batch = np.array_split(index_epoch[:size-batch_left],num_iter)

    # the remaining index
    if batch_left:
        index_batch.append(index_epoch[size-batch_left:])

    return index_batch


class Processor:
    """Creates a class to execute model training, validation and test"""

    model: VitAug
    data_train: Tuple[np.ndarray, np.ndarray]
    data_val: Tuple[np.ndarray, np.ndarray]
    data_test: Tuple[np.ndarray, np.ndarray]

    losses_train: list | None
    accuracies_train: list | None
    times_train: list | None
    losses_val: list | None
    accuracies_val: list | None
    times_val: list | None
    accuracy_test: int | None

    def __init__(
            self,
            model: VitAug | str,
            data_train: Tuple[np.ndarray, np.ndarray] | None,
            data_val: Tuple[np.ndarray, np.ndarray] | None,
            data_test: Tuple[np.ndarray, np.ndarray] | None,
            learning_rate: float | None,
            batch_size: float | None,
            num_epochs: float | None,

    ):
        """
        Initialize the processor.

        Args:
            model (VitAug): the vit model with augmentation method
            data_train (tuple | None): training dataset, (images, labels). Can be None if training and validation will not be executed.
            data_val (tuple | None): validation dataset (images, labels). Can be None if validation will not be executed.
            data_test (tuple | None): test dataset (images, labels). Can be None if test will not be executed.
            learning_rate (float | None): learning rate during training. Can be None if training and validation will not be executed.
            batch_size (int | None): size of the batch under each epoch. Can be None if training and validation will not be executed.
            num_epochs (int | None): number of epochs. Can be None if training and validation will not be executed.
        """
        if type(model) is str:
            self.model = tf.keras.models.load_model(model)
        else:
            self.model = model

        # datasets
        self.data_train = data_train
        self.data_val = data_val
        self.data_test = data_test

        # training related parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self._add_metrics()

    def train(
            self,
            monitor_on_validation: bool | None = True,
            path_root: str | None = None,
            save_period: int | None = None,
            save_model: bool | None = False,
            save_log: bool | None = False,
    ):
        """
        Train the model

        Args:
            monitor_on_validation (bool): If True, the model will be assessed on the validation set for each epoch. Default to True
            path_root (str | None): root path to save the models and log. Default to None
            save_period (int | None): If not None, the model will be saved every save_period of epochs. The saved filename is
                'epochs {} - {}.h5'.format(epochs, path_root). Default to None.
            save_model (bool): If True, save the final model after training. The saved filename is f'{path_root}.h5'. Default to False.
            save_log (bool): If True, save the log after training. The saved filename is f'{path_root}.json'. Default to False.
        """
        if path_root is None and (save_period is not None or save_log or save_model):
            raise ValueError(f"Incompatible values : filename is None but save_period is {save_period}, save_model is {save_model}, and save_log is {save_log}")
        if path_root is not None and save_period is None and not save_log and not save_model:
            warn(f'path_root is given the value {path_root}, but save_period is {save_period}, save_model is {save_model}, and save_log is {save_log}')

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # begin training over epochs
        print(f'start: training, with sampling method {self.model.aug.sampling_method}')
        for e in range(self.num_epochs):
            loss_train, accuracy_train, time_spent_train = self._training(optimizer, self.batch_size)

            self.losses_train.append(loss_train)
            self.accuracies_train.append(accuracy_train)
            self.times_train.append(time_spent_train)

            if monitor_on_validation:
                # evaluate the model on the validation data set
                loss_val, accuracy_val, time_spent_val = self._reference(self.data_val)
                self.losses_val.append(loss_val)
                self.accuracies_val.append(accuracy_val)
                self.times_val.append(time_spent_val)

                # note: the accuracy on train, is lambda * accuracy(preds, labels1) + (1-lambda) * accuracy(preds,labels2)
                # which is different from the accuracy on the validation data set
                print('''Epoch {}, train:: time: {:.3f}s, loss: {:.3f}, aug_accuracy: {:.3f}% \
                    --- test:: time: {:.3f}s, loss: {:.3f}, accuracy: {:.3f}%'''.format(
                    e+1, time_spent_train, loss_train, accuracy_train*100,
                    time_spent_val, loss_val, accuracy_val*100))
                
            else:
                print('''Epoch {}, train:: time: {:.3f}s, loss: {:.3f}, aug_accuracy: {:.3f}%'''.format(
                    e+1, time_spent_train, loss_train, accuracy_train*100))
                
            if save_period is not None and (e+1) % save_period == 0:
                self.model.save('epochs {} - {}.h5'.format(e+1, path_root))
        
        print(f'finish: training, with sampling method {self.model.aug.sampling_method}')
                
        if save_model:
            self.model.save(path_root + '.h5')

        if save_log:
            self.save_log(path_root + '.json')

    def test(self):
        """
        Test the model on hold-out test set
        
        Returns:
            accuracy_test (float): accuracy on the test set.
        """

        print('start: test')
        _, accuracy_test, _ = self._reference(self.data_test)
        self.accuracy_test = accuracy_test
        print('finish: test. Accuracy is {:.3f}%'.format(accuracy_test * 100))
        
        return accuracy_test
    
    def reset_model(self, model = VitAug | str):
        """
        Reset the model
        
        Args:
            model (VitAug): the model to reset
        """
        if type(model) is str:
            self.model = tf.keras.models.load_model(model)
        else:
            self.model = model

        self._add_metrics()

    def save_log(self,  filename: str):
        """
        Save the log
        
        Args:
            filename (str): path to save the log
        """
        log = {
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'losses_train': self.losses_train,
            'accuracies_train': self.accuracies_train,
            'times_train': self.times_train,
            'losses_val': self.losses_val,
            'accuracies_val': self.accuracies_val,
            'times_val': self.times_val,
            'accuracy_test': self.accuracy_test
        }

        with open(filename, 'w') as f:
            json.dump(log, f, indent=4)

    
    def _add_metrics(self):
        """Add metrics-related attributes"""
        # metrics
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.accuracy_object = tf.keras.metrics.SparseCategoricalAccuracy()

        # metrics records
        self.losses_train = []
        self.accuracies_train = []
        self.times_train = []
        self.losses_val = []
        self.accuracies_val = []
        self.times_val = []
        self.accuracy_test = None

    def _training(self, optimizer: Optimizer, batch_size: int):
        """
        Model training for one epoch
        
        Args:
            optimizer (Optimizer): Optimizer to train the model
            batch_size (int): size of the batch under each epoch 

        Returns:
            loss_train (float): loss of this epoch
            accuracy_train (float): augmentation accuracy of this epoch, defined as 
                lambda * accuracy(preds, labels1) + (1-lambda) * accuracy(preds,labels2)
            time_spent_train (float): the time spent on trainng for this epoch, in seconds
        """
        losses_train_batch = []
        accuracies_train_batch = []
        # train the model for one epoch
        t1 = time.time()
        index_batch = index_batch_generate(self.data_train[0].shape[0], batch_size)
        for index in index_batch:
            images_batch = self.data_train[0][index]
            labels_batch = self.data_train[1][index].astype(float)
            
            # forwards
            with tf.GradientTape() as tape:
                preds = self.model(images_batch, training=True)

                loss = self.model.aug.lam * self.loss_object(labels_batch, preds) + \
                    (1-self.model.aug.lam) * self.loss_object(labels_batch[self.model.aug.index.numpy()],preds)
                
                accuracy = self.model.aug.lam * self.accuracy_object(labels_batch, preds) + \
                    (1-self.model.aug.lam) * self.accuracy_object(labels_batch[self.model.aug.index.numpy()],preds)

            # backwords
            gradients = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            # add loss and accuracy for this epoch
            losses_train_batch.append(loss)
            accuracies_train_batch.append(accuracy)

        t2 = time.time()

        loss_train = np.mean(losses_train_batch)
        accuracy_train = np.mean(accuracies_train_batch)
        time_spent_train = t2-t1

        return float(loss_train), float(accuracy_train), time_spent_train
    

    def _reference(self, dataset: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, float, float]:
        """
        Model reference

        Args:
            dataset (tuple | None): dataset (images, labels) to execute reference on.
        
        Returns:
            loss_reference (float): loss on the set
            accuracy_reference (float): accuracy on the set (without augmentation)
            time_spent_reference (float): time spent on reference, in seconds
        """
        t1 = time.time()
        preds_reference = self.model.predict(dataset[0], verbose=0)
        t2 = time.time()

        loss_reference = self.loss_object(dataset[1], preds_reference)
        accuracy_reference = self.accuracy_object(dataset[1], preds_reference)
        time_spent_reference = t2-t1

        return float(loss_reference), float(accuracy_reference), time_spent_reference

