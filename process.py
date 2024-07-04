import tensorflow as tf
import time
import numpy as np
from models import VitAug


def index_batch_generate(size:int, batch:int):
    '''generate index for each batch withn an epoch
    
    Args:
        size: size of the data wihin one epoch
        bach: size of the batch
    
    Returns:
        a list of indices
    '''
    assert size > batch, f'batch {batch} is bigger than data size {size}'

    num_iter = size // batch
    batch_left = size % batch
    index_epoch = np.random.choice(size,size,replace=False)
    index_batch = np.array_split(index_epoch[:size-batch_left],num_iter)

    # the remaining index
    if batch_left:
        index_batch.append(index_epoch[size-batch_left:])

    return index_batch


def model_train_val(dataset : tuple,  
                    epoch : int, 
                    batch : int, 
                    learning_rate : float, 
                    model_param : dict, 
                    filename : str | None = None):
    '''train the ViT with MixUp, and monitor accuracy on validation set for each epoch

    Args:
        dataset: ((images_train,labels_train), (images_val,labels_val))
        epoch: number of epochs
        batch: size of the batch
        learning_rate: learning rate
        model_param: a dict with values for the parameters of the model
            sampling_method : method to generate lambda. 'beta' indicates beta, 'uniform' indicates uniform
            image_size: width or height of the input images
            patch_size: width or height of the size
            num_classes: number of the classes
            dim: length of the word vector
            heads: number of heads of attention block
            mlp_dim: dimension of the hiddin layer of the MLP
            dropout: dropout percentage
            alpha: float, parameter for beta distribution
            uniform_range: tuple, predefined range to generate lambda uniformly 
        filename: path to save the model. If None, the model will not be saved

    Returns:
        ((train_loss_record,train_accuracy_record,train_time_record),
        (val_loss_record,val_accuracy_record,val_time_record))
    '''

    (images_train, labels_train), (images_val,labels_val) = dataset
    images_train = images_train.astype(float) / 255.0
    images_val = images_val.astype(float) / 255.0

    size = images_train.shape[0]
    model = VitAug(**model_param)
    
    # set optiizer and metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    accuracy_object = tf.keras.metrics.SparseCategoricalAccuracy()

    # to record loss, accuracy and time
    train_loss_record = []
    train_accuracy_record = []
    train_time_record = []

    val_loss_record = []
    val_accuracy_record = []
    val_time_record = []
    
    print('\nmethod = ' + model_param['sampling_method'] +'::')
    print('----------start : train and monitor------------')
    for e in range(epoch):

        # metrics
        train_loss = []
        train_accuracy = []

        # train the model
        t1 = time.time()
        index_batch = index_batch_generate(size,batch)
        for index in index_batch:
            images_batch = images_train[index]
            labels_batch = labels_train[index].astype(float)
            
            # forwards
            with tf.GradientTape() as tape:
                preds = model(images_batch, training=True)

                loss = model.aug.lam * loss_object(labels_batch, preds) + \
                    (1-model.aug.lam) * loss_object(labels_batch[model.aug.index.numpy()],preds)
                accuracy = model.aug.lam * accuracy_object(labels_batch, preds) + \
                    (1-model.aug.lam) * accuracy_object(labels_batch[model.aug.index.numpy()],preds)

            # backwords
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # add loss and accuracy for this epoch
            train_loss.append(loss)
            train_accuracy.append(accuracy)

        t2 = time.time()

        # motitor metrics on val set
        val_preds_raw = model.predict(images_val,verbose=0)
        t3 = time.time()
        val_loss = loss_object(labels_val,val_preds_raw)
        val_preds = np.expand_dims(np.argmax(val_preds_raw,1),1)
        val_accuracy = np.mean(np.equal(val_preds, labels_val))

        # note: the accuracy on train, aug_accuracy, 
        # is lambda * accuracy(preds, labels1) + (1-lambda) * accuracy(preds,labels2)
        print('''Epoch {}, train:: time: {:.3f}s, loss: {:.3f}, aug_accuracy: {:.3f}% \
              --- test:: time: {:.3f}s, loss: {:.3f}, accuracy: {:.3f}%'''.format(
            e+1, t2-t1, np.mean(train_loss), np.mean(train_accuracy)*100,
            t3-t2, val_loss, val_accuracy*100))
        
        # record metrics
        train_loss_record.append(np.mean(train_loss))
        train_accuracy_record.append(np.mean(train_accuracy))
        train_time_record.append(t2-t1)

        val_loss_record.append(val_loss)
        val_accuracy_record.append(val_accuracy)
        val_time_record.append(t3-t2)
        

    print('----------finish : train and monitor------------\n')

    if filename is not None:
        model.save(filename)

    return ((train_loss_record,train_accuracy_record,train_time_record),
            (val_loss_record,val_accuracy_record,val_time_record))


def model_test(filename : str, images_test : np.ndarray, labels_test : np.ndarray):
    '''evaluate performace on the holdout test set
    
    Args : 
        filename : path of the saved model
        images_test : images in the hold out test set
        labels_test : labels in the hold out test set

    Returns : 
        (test_loss, test_accuracy) : two scalers
    '''
    images_test = images_test.astype(float) / 255.0

    print('----------start : test on hold out set------------')
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model = tf.keras.models.load_model(filename)
    test_preds_raw = model.predict(images_test,verbose=0)

    test_loss = loss_object(labels_test,test_preds_raw)
    test_preds = np.expand_dims(np.argmax(test_preds_raw,1),1)
    test_accuracy = np.mean(np.equal(test_preds, labels_test))
    print('----------finish : test on hold out set------------\n')

    return (test_loss,test_accuracy)