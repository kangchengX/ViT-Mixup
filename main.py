import argparse
import tensorflow as tf
import numpy as np
from data_utilis import spilt_data
from models import VitAug
from process import Processor
from display_tools import report_summary
from datetime import datetime
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # data related
    parser.add_argument('--ratio_dev', type=float, default=0.8,
                        help="Ratio for development set (i.e. training and validation) in the whole data set. Default to 0.8.")
    parser.add_argument('--ratio_train', type=float, default=0.9,
                        help="Ratio for train set in the whole development set. Default to 0.9.")

    # model structure related
    parser.add_argument('--sampling_method', type=str, choices=['beta', 'uniform'], default='uniform',
                        help="Method to generate lambda. 'beta' indicates beta, 'uniform' indicates uniform. Default to 'uniform'.")
    parser.add_argument('--image_size', type=int, default=32,
                        help="Width or height of input images. Default to 32.")
    parser.add_argument('--patch_size', type=int, default=4,
                        help="Width or height of patches. Default to 4.")
    parser.add_argument('--num_classes', type=int, default=10,
                        help="Number of the classes")
    parser.add_argument('--dim', type=int, default=256,
                        help="Dimension of the word vectors. Default to 256.")
    parser.add_argument('--depth', type=int, default=8,
                        help="Number of transformer blocks. Default to 8.")
    parser.add_argument('--num_heads', type=int, default=8,
                        help="Number of heads in the transformer. Default to 8.")
    parser.add_argument('--mlp_dim', type=int, default=512,
                        help="Hiddin dimension of mlp blocks. Default to 512.")
    parser.add_argument('--dropout', type=float, default=0.5,
                        help="Dropout percentage. Default to 0.5")
    parser.add_argument('--alpha', type=float,
                        help="Parameter for beta distribution (used if sampling_method is 'beta'). Default to None")
    parser.add_argument('--uniform_range', type=float, nargs=2, default=(0.0, 1.0),
                        help="Predefined range to generate lambda uniformly (used if sampling_method is 'uniform'). Default to (0.0, 1.0)")
    
    # optimization parameters in training
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate durning training. Default to 0.001')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size durning training. Default to 64.')
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='Number of epochs durning training. Default to 40.')

    # execution related
    parser.add_argument('--monitor_on_validation', action='store_false',
                        help='Indicates if assess model on the validation set durning training. Default to True')

    # files creating related
    parser.add_argument('--path_root', type=str, default=datetime.now().strftime(r"%Y-%m-%d %H-%M-%S"),
                        help='Path root to save models and log if not None. Default to current time.')
    parser.add_argument('--save_model', action='store_true',
                        help='Indicates if save the final model. path_root should not be None if this is True. Default to False.')
    parser.add_argument('--save_period', type=int,
                        help='Save the model every save_period of epochs if not None. path_root should not be None if this is not None. Default to None.')
    parser.add_argument('--save_log', action='store_false',
                        help='Indicates if log will be saved. path_root should not be None if this is not True. Default to True.')

    args = parser.parse_args()

    # load data
    (images_train, labels_train), (images_test, labels_test) = tf.keras.datasets.cifar10.load_data()
    images = np.concatenate((images_train, images_test), axis=0)
    labels = np.concatenate((labels_train,labels_test), axis=0)
    data_train, data_val, data_test = spilt_data((images, labels))

    # initialize model
    model = VitAug(
        sampling_method=args.sampling_method,
        alpha = args.alpha,
        uniform_range = args.uniform_range,
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=args.num_classes,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout
    )

    print(model)

    # execute differenct processes
    processor = Processor(
        model=model,
        data_train=data_train,
        data_val=data_val,
        data_test=data_test,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs
    )

    processor.train(
        path_root=args.path_root, 
        save_period= args.save_period,
        monitor_on_validation=args.monitor_on_validation,
        save_log=args.save_log
    )

    processor.test()


    print('**************summary*************')
    print('''-----results on developmennt set------''')
    report_summary((processor.losses_train, processor.accuracies_train, processor.times_train), 'train')
    report_summary((processor.losses_val, processor.accuracies_val, processor.times_val), 'validation')

    print('''-----results on hold-out test set----''')
    report_summary((processor.accuracy_test), 'test')


    
