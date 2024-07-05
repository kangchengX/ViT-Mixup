import tensorflow as tf
import display_tools
from data_utilis import spilt_data
from process import model_train_val, model_test

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

(images_train, labels_train), (images_test, labels_test) = tf.keras.datasets.cifar10.load_data()

# split the dataset
dataset = spilt_data((images_train, labels_train))
dataset_dev = (dataset[0],dataset[1])

#visualize the mixup using 16 images
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
display_tools.visualise_mixup(images=images_train[0:16], 
                              size=(4,4), 
                              sampling_method='1', 
                              alpha=1.0,
                              filename='mixup.png')


# two model configurations
model_param1 = {
    'sampling_method': 'beta',
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
    'sampling_method': 'uniform',
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

model_params = {model_param1['sampling_method'] : model_param1,
                model_param2['sampling_method'] : model_param2}

methods = model_params.keys()

# records metrics, 
# {method : (dev_record, test_record)}, 
# devcord : ((train_metric lists),(test_metric lists))
# test_record : (train_metric scalers)
records = {}

for method in methods:
    record_dev = model_train_val(dataset=dataset_dev,
                                 filename='saved-model'+'-'+method,
                                 epoch=40, 
                                 batch=128, 
                                 learning_rate=0.001,
                                 model_param=model_params[method])
    record_holdout = model_test(filename='saved-model'+'-'+method,
                                images_test=dataset[2][0],
                                labels_test=dataset[2][1])
    
    records[method] = (record_dev,record_holdout)

print('**************summary*************')

print('''-----results on developmennt set------''')
for method in methods:
    print('method = ' + method +'::')
    display_tools.report_summary(records[method][0][0],'train')
    display_tools.report_summary(records[method][0][1],'validation')

print('''-----results on hold out test set----''')
for method in methods:
    print('method = ' + method +'::')
    display_tools.report_summary(records[method][1],'test')