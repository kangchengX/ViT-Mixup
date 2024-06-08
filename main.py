import tensorflow as tf
import numpy as np
from PIL import Image
from networks import MixUp, VitAug
import time

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.random.set_seed(197)


def mixup_visualiation(filename:str, images:tf.Tensor, size:tuple, sampling_method:str, **kwargs):
    '''visualise the mixup results with a montage
    
    Args :
        filename : path to save the montage
        images : image examples
        size : size of the montage, (rows, cols)
        sampling_method : method to generate lambda. '1' indicates beta, '2' indicate uniform
        alpha: float, parameter for beta distribution
        uniform_range: tuple, predefined range to generate lambda uniformly
    '''

    assert images.shape[0] == size[0]*size[1]

    images = tf.cast(images,dtype=np.float32)

    mixup_layer = MixUp(sampling_method,**kwargs)
    images = mixup_layer(images,training=True)

    # generate and save montage
    images = tf.concat([tf.concat([images[i+4*j,...] for i in range(size[1])],1) for j in range(size[0])],0)
    images = Image.fromarray(images.numpy().astype(np.uint8))
    images.save(filename)

    
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


def model_train_test(dataset:tuple, filename:str, 
                     epoch:int, batch:int, learning_rate:float, 
                     model_param:dict):
    '''train the ViT with MixUp, and monitor accuracy on test set for each epoch

    Args:
        dataset: ((images_train,labels_train), (images_test,labels_test))
        filename: path to save the model
        epoch: number of epochs
        batch: size of the batch
        learning_rate: learning rate
        model_param: a dict with values for the parameters of the model
            sampling_method : method to generate lambda. '1' indicates beta, '2' indicate uniform
            image_size: width or height of the input images
            patch_size: width or height of the size
            num_classes: number of the classes
            dim: length of the word vector
            heads: number of heads of attention block
            mlp_dim: dimension of the hiddin layer of the MLP
            dropout: dropout percentage
            alpha: float, parameter for beta distribution
            uniform_range: tuple, predefined range to generate lambda uniformly 

    Returns:
        ((train_loss_record,train_accuracy_record,train_time_record),
        (test_loss_record,test_accuracy_record,test_time_record))
    '''

    (images_train, labels_train), (images_test,labels_test) = dataset
    images_train = images_train.astype(float) / 255.0
    images_test = images_test.astype(float) / 255.0

    size = images_train.shape[0]
    model = VitAug(**model_param)
    
    # set optiizer and metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    accuracy_object = tf.keras.metrics.SparseCategoricalAccuracy()
    train_loss = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.Mean()

    # to record loss, accuracy and time
    train_loss_record = []
    train_accuracy_record = []
    train_time_record = []

    test_loss_record = []
    test_accuracy_record = []
    test_time_record = []
    
    print('\nmethod = ' + model_param['sampling_method'] +'::')
    print('----------start : train and monitor------------')
    for e in range(epoch):

        # reset states of metrics
        train_loss.reset_states()
        train_accuracy.reset_states()

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
            train_loss(loss)
            train_accuracy(accuracy)

        t2 = time.time()

        # motitor metrics on test set
        test_preds_raw = model.predict(images_test,verbose=0)
        t3 = time.time()
        test_loss = loss_object(labels_test,test_preds_raw)
        test_preds = np.expand_dims(np.argmax(test_preds_raw,1),1)
        test_accuracy = np.mean(np.equal(test_preds, labels_test))

        # note: the accuracy on train, aug_accuracy, 
        # is lambda * accuracy(preds, labels1) + (1-lambda) * accuracy(preds,labels2)
        print('''Epoch {}, train:: time: {:.3f}s, loss: {:.3f}, aug_accuracy: {:.3f}% --- test:: time: {:.3f}s, loss: {:.3f}, accuracy: {:.3f}%'''.format(
            e+1, t2-t1, train_loss.result(), train_accuracy.result()*100,
            t3-t2, test_loss, test_accuracy*100))
        
        # record metrics
        train_loss_record.append(train_loss.result())
        train_accuracy_record.append(train_accuracy.result())
        train_time_record.append(t2-t1)

        test_loss_record.append(test_loss)
        test_accuracy_record.append(test_accuracy)
        test_time_record.append(t3-t2)
        

    print('----------finish : train and monitor------------\n')

    if filename is not None:
        model.save(filename+'-'+model_param['sampling_method'])

    return ((train_loss_record,train_accuracy_record,train_time_record),
            (test_loss_record,test_accuracy_record,test_time_record))


def results_visualisation(filename_model:str, filename_fig:str, images, labels, class_names):
    '''visualise results of the trained model by saving the montage and printing the ture and predicted labels
    
    Args: 
        filename_model: path of the saved model
        filename_fig: path to save the montage
        images: a tensor with shape (num, height, width, chanels)
        labels: labels of the images
        class_name: true names of the classes
    
    '''

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


def spilt_data(data:tuple, ratio_dev=0.8, ratio_train=0.9):
    '''split data to train set, validation set and hold out test set

    Args :
        data : (images,labels)
        ratio_dev : ratio for development set (i.e. train and validation) in the whole data set
        ratio_dev : ratio for train set in the whole development set

    Returns :
        dataset :  ((images_train,labels_train), (images_val,labels_val), (images_test,labels_test))
    '''

    # split dataset to development and hold out set
    images,labels = data
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

    dataset = ((images_train,labels_train), (images_val,labels_val), (images_test,labels_test))

    return dataset


def report_summary(record:tuple, dataset_type:int):
    '''report summary of results of the model
    Args: 
        record: (loss_array, accuracy_array, time_array) for train and validation
            (loss_scaler,accuracy_scaler,time_scaler) for hold out test
        dataset_type: train, validation or test
    '''

    if dataset_type == 'test':
       print('''\ton hold out test set:
            loss: {:.3f}
            accuracy: {:.3f}%
        '''.format(record[0],record[1]*100))
    
    elif dataset_type in ['train','validation']:
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
        raise ValueError('unknown dataset type : ' + dataset_type)
    

def model_test(filename:str, sampling_method:str, images_test:np.ndarray, labels_test:np.ndarray):
    '''evaluate performace on the holdout test set
    
    Args : 
        filename, sampling_method : filename+'-'_samping method is the path of the saved model
        images_test : images in the hold out test set
        labels_test : labels in the hold out test set

    Returns : 
        (test_loss, test_accuracy) : two scalers  
    '''
    images_test = images_test.astype(float) / 255.0

    print('----------start : test on hold out set------------')
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model = tf.keras.models.load_model(filename+'-'+sampling_method)
    test_preds_raw = model.predict(images_test,verbose=0)

    test_loss = loss_object(labels_test,test_preds_raw)
    test_preds = np.expand_dims(np.argmax(test_preds_raw,1),1)
    test_accuracy = np.mean(np.equal(test_preds, labels_test))
    print('----------finish : test on hold out set------------\n')

    return (test_loss,test_accuracy)



if __name__ == '__main__':

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    mixup_visualiation('mixup.png',train_images[0:16],(4,4),'1',alpha = 1.0)

    model_param = {
        'sampling_method': '1',
        'image_size': 32,
        'patch_size': 4,
        'num_classes': 10,
        'dim': 256,
        'depth': 8,
        'heads': 8,
        'mlp_dim': 512,
        'dropout': 0.2,
        'alpha': 1.0
    }

    model_train_test(dataset=((train_images,train_labels),(test_images,test_labels)),
                     filename='saved-model',
                     epoch=20,
                     batch=128,
                     learning_rate=0.001,
                     model_param=model_param)

    # note, the filename_model should be 'saved-model-2' for method = '2'
    results_visualisation(filename_model='saved-model-1',
                          filename_fig='result.png',
                          images=train_images[0:36,...],
                          labels=train_labels[0:36],
                          class_names=class_names)
    

    # split the dataset
    dataset = spilt_data((train_images,train_labels))
    dataset_dev = (dataset[0],dataset[1])


    # two model configurations
    model_param1 = {
        'sampling_method': '1',
        'image_size': 32,
        'patch_size': 4,
        'num_classes': 10,
        'dim': 256,
        'depth': 8,
        'heads': 8,
        'mlp_dim': 512,
        'dropout': 0.2,
        'alpha': 1.0
    }

    model_param2 = {
        'sampling_method': '2',
        'image_size': 32,
        'patch_size': 4,
        'num_classes': 10,
        'dim': 256,
        'depth': 8,
        'heads': 8,
        'mlp_dim': 512,
        'dropout': 0.2,
        'uniform_range': (0.0,1.0)
    }

    model_params = {'1':model_param1,'2':model_param2}
    methods = ['1','2']

    # records metrics, 
    # {method : (dev_record, test_record)}, 
    # devcord : ((train_metric lists),(test_metric lists))
    # test_record : (train_metric scalers)
    records = {}

    for method in methods:
        record_dev = model_train_test(dataset=dataset_dev,
                                          filename='saved-model',
                                          epoch=20, batch=128, learning_rate=0.001,
                                          model_param=model_params[method])
        record_holdout = model_test(filename='saved-model',
                                    sampling_method=method,
                                    images_test=dataset[2][0],
                                    labels_test=dataset[2][1])
        
        records[method] = (record_dev,record_holdout)

    print('**************summary*************')

    print('''-----results on developmennt set------''')
    for method in methods:
        print('method = ' + method +'::')
        report_summary(records[method][0][0],'train')
        report_summary(records[method][0][1],'validation')

    print('''-----results on hold out test set----''')
    for method in methods:
        print('method = ' + method +'::')
        report_summary(records[method][1],'test')