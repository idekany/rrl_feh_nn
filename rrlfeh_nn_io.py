import argparse
import os

default_parameter_file = '@gfeh_train.par'


def argparser():
    """
    Creates an argparse.ArgumentParser object for reading in parameters from a file.
    :return:
    """
    ap = argparse.ArgumentParser(fromfile_prefix_chars='@',
                                 description='Train and deploy a deep-learned [Fe/H] estimator'
                                             'based on Gaia time-series photometry.',
                                 epilog="")

    # use custom line parser for the parameter file
    ap.convert_arg_line_to_args = convert_arg_line_to_args

    # EXECUTION MODE
    # --------------

    ap.add_argument('-t',
                    '--train',
                    action='store_true',
                    help='Train the model.')

    ap.add_argument('-p',
                    '--predict',
                    action='store_true',
                    help='Deploy a trained model to make predictions.')

    ap.add_argument('-r',
                    '--refit',
                    action='store_true',
                    help='Retrain themodel on the entire development (training + validation) '
                         'set using the best hyper-parameters.')

    ap.add_argument('-cv',
                    '--cross_validate',
                    action='store_true',
                    help='Use cross-validation with the development set, and find the best '
                         'hyper-parameters if a nested list is provided for --hparam_grid. '
                         'NOTE: if unsetting this parameter, the --hparam_grid argument '
                         'is expected to have a single element.')

    ap.add_argument('--nn_type',
                    action='store',
                    type=str,
                    choices=['rnn', 'cnn'],
                    default='rnn',
                    help='The architecture type of the neural network.')

    ap.add_argument('-cpu',
                    action='store_true',
                    help='Use CPU only.')

    # I/O PARAMETERS
    # --------------

    ap.add_argument('-v',
                    '--verbose',
                    action='store_true',  # assign True value if used
                    help='Generate verbose output.')

    ap.add_argument('--seed',
                    action='store',
                    type=int,
                    default=1,
                    help='Integer seed for reproducible results.')

    ap.add_argument('--rootdir',
                    action='store',
                    type=str,
                    default=os.path.expanduser('~'),
                    help='Full path of the root directory '
                         '(all other directory and file names will be relative to this).')

    ap.add_argument('--outdir',
                    action='store',
                    type=str,
                    default=os.path.expanduser('results'),
                    help='Path of the output directory (relative to root).')

    ap.add_argument('--input_model_dir',
                    action='store',
                    type=str,
                    default='.',
                    help='Path of the directory of the input model files(s) for prediction (relative to root).')

    ap.add_argument('--lcdir',
                    action='store',
                    nargs='+',
                    type=str,
                    default='lc',
                    help='Relative path of the directory containing the input light curves.')

    ap.add_argument('--input_file',
                    action='store',
                    type=str,
                    required=True,
                    help='The file containing the metadata of the stars.')

    ap.add_argument('--target_output_file',
                    action='store',
                    type=str,
                    default='target.out',
                    help='The file containing the metadata of the stars.')

    ap.add_argument('--wavebands',
                    action='store',
                    nargs='+',  # we expect a flexinble number of values, but at least one value
                    type=str,
                    choices=['k', 'g', 'bp', 'rp'],
                    default='g',
                    help='Input waveband(s).')

    ap.add_argument('--lcfile_suffices',
                    action='store',
                    nargs='*',
                    type=str,
                    default='.dat',
                    help='Suffices of the light-curve files (must match the --wavebands arguments).')

    ap.add_argument('--progress_plot_subdir',
                    action='store',
                    type=str,
                    default='progress_plots',
                    help='Subdirectory for the plots of the training progress.')

    ap.add_argument('--predict_train_output',
                    action='store',
                    type=str,
                    default='predict_train',
                    help='Filename suffix for the predictions on the training data.')

    ap.add_argument('--predict_val_output',
                    action='store',
                    type=str,
                    default='predict_val',
                    help='Filename suffix for the predictions on the validation data.')

    ap.add_argument('--predict_test_output',
                    action='store',
                    type=str,
                    default='predict_test',
                    help='Filename suffix for the predictions on the test data.')

    ap.add_argument('--liveplotname',
                    action='store',
                    type=str,
                    default='liveplot',
                    help='Filename for the live progress plot during training.')

    ap.add_argument('--plot_input_data',
                    action='store_true',
                    help='Plot the input time series (and period-amplitude diagram).')

    ap.add_argument('--model_file_prefix',
                    action='store',
                    type=str,
                    default='model',
                    help='File for saving/loading the model architecture.')

    ap.add_argument('--weights_file_prefix',
                    action='store',
                    type=str,
                    default='model',
                    help='File for saving/loading the model weights.')

    ap.add_argument('--metascaler_file',
                    action='store',
                    type=str,
                    default='scaler',
                    help='File for saving/loading the standard scaling coefficients of the metadata.')

    ap.add_argument('--save_model',
                    action='store_true',
                    help='Save the (best) trained model.')

    ap.add_argument('--save_checkpoints',
                    action='store_true',
                    help='Save model weights at checkpoints.')

    ap.add_argument('--log_training',
                    action='store_true',
                    help='Save log of training progress.')

    ap.add_argument('--plot_prediction',
                    action='store_true',
                    help='Plot the predictions vs ground truth on the training and validation data.')

    # DATA PARAMETERS
    # ---------------

    ap.add_argument('--nbins',
                    action='store',
                    type=int,
                    default=None,
                    help='Number of phase bins to compute for the input light curve if `nn_type` is `rnn`. '
                         'If provided, an equidistant grid of phase bins will be used between phases 0 '
                         'and `max_phase`. If not provided, the data will not be binned.')

    ap.add_argument('--max_phase',
                    action='store',
                    type=float,
                    default=1.0,
                    help='The maximum phase value to be considered for the light curves if `nn_type` is `rnn`. '
                         'All input data beyond this phase value will be truncated.')

    ap.add_argument('--n_aug',
                    action='store',
                    type=int,
                    default=None,
                    help='Number of augmented time series per object. If None, no augmented data are used.'
                         'This feature is currently supported in CNN mode only.')

    ap.add_argument('--columns',
                    action='store',
                    type=str,
                    nargs='+',
                    default=['id', 'period'],
                    help='The list of names of the columns to be read in from the input file.')

    ap.add_argument('--features',
                    action='store',
                    type=str,
                    nargs='+',
                    default=['period', 'FeH'],
                    help='The list of features (including the target variable).')

    ap.add_argument('--subset',
                    action='store',
                    type=str,
                    nargs='*',
                    help='Expression for subsetting the input data, passed as the '
                         'first parameter of pandas.DataFrame.query()')

    ap.add_argument('--meta_input',
                    action='store',
                    type=str,
                    nargs='+',
                    help='The list of the names of the features to be used on a second input layer of the model.'
                         'If not provided, then only one input layer will be used (i.e., the one for the sequence '
                         'data). In either case, please make sure to select a model that has the appropriate '
                         'input layers implemented.')

    ap.add_argument('--explicit_test_frac',
                    action='store',
                    type=float,
                    help='If provided, the specified fraction of the data will be used '
                         'as an explicit, held-out test set.')

    ap.add_argument('--weighing_by_density',
                    action='store',
                    type=float,
                    help="If provided, sample weights will be assigned by the inverse "
                         "of the data's number density as estimated by KDE and up to the "
                         "density value specified by this parameter.")

    # TRAINING PARAMETERS
    # -------------------

    ap.add_argument('--eval_metric',
                    action='store',
                    type=str,
                    choices=['r2', 'rmse', 'wrmse', 'mae', 'wmae', 'medae'],
                    default='r2',
                    help='Model selection metric.')

    ap.add_argument('--k_fold',
                    action='store',
                    type=int,
                    default=10,
                    help='Number of cross-validation folds.')

    ap.add_argument('--ensemble',
                    action='store_true',
                    help='If True, a k-fold model ensemble will be created.')

    ap.add_argument('--split_frac',
                    action='store',
                    type=float,
                    default=0.1,
                    help='CV fraction in stratified shuffle-split if k_fold < min_folds_cv '
                         '(otherwise k-fold CV is done)')

    ap.add_argument('--n_repeats',
                    action='store',
                    type=int,
                    default=1,
                    help='Number of repeats in the cross-validation.')

    ap.add_argument('--pick_fold',
                    type=int,
                    action='store',
                    nargs='+',
                    help='Pick (a) specified fold(s) from the k_fold folds and use it/them for CV.'
                         'This makes sense for doing a k-fold CV in separate runs for each fold '
                         'by using the same randomization.')

    ap.add_argument('--n_epochs',
                    action='store',
                    type=int,
                    default=100,
                    help='Number of training epochs.')

    ap.add_argument('--auto_stop',
                    action='store',
                    type=str,
                    default=None,
                    choices=['early', 'late'],
                    help="late/early: stop the training if the training/validation loss does not decrease "
                         "by more than 'min_delta' within 'patience' number of epochs.")

    ap.add_argument('--min_delta',
                    action='store',
                    type=float,
                    default=10e-5,
                    help="Minimum decrease in loss required by 'auto_stop'.")

    ap.add_argument('--patience',
                    action='store',
                    type=int,
                    default=1000,
                    help="Patience parameter required by 'auto_stop'.")

    ap.add_argument('--batch_size_per_replica',
                    action='store',
                    type=int,
                    default=64,
                    help="The batch size used by the optimization algorithm. "
                         "If using multiple devices with mirrored strategy, "
                         "then each replica will work with this batch size.")

    ap.add_argument('-lr',
                    '--learning_rate',
                    action='store',
                    type=float,
                    default=0.01,
                    help="The learning rate of the optimization algorithm.")

    ap.add_argument('--n_zoom',
                    action='store',
                    type=int,
                    default=200,
                    help="Number of epochs after which to zoom in on the progress plot.")

    ap.add_argument('--n_update',
                    action='store',
                    type=int,
                    default=100,
                    help="The progress plot will be updated after every 'n_update' epochs.")

    ap.add_argument('--optimize_lr',
                    action='store_true',
                    help="Run training for the optimization of the learning rate. "
                         "If True the learning rate will be changed on every epoch "
                         "according to min_learning_rate and lr_increment_coeff (see source code), "
                         "and at each epoch, min_learning_rate will be multiplied by "
                         "10 ** (epoch * lr_increment_coeff)"
                         "Finally, a loss vs learning_rate plot will be returned.")

    ap.add_argument('--decay',
                    action='store',
                    type=float,
                    default=0.0,
                    help="Learning rate decay: "
                         "decay = ( L_(t-1)/L_t - 1 ) / batches_per_epoch"
                         "where L_t is the learning rate at epoch t;"
                         "e.g., decay = 0.00008 corresponds to L_t/L_(t-1) ~ 0.99")

    ap.add_argument('--hpars',
                    required=True,
                    action='append',
                    nargs='*',
                    help="The hyper-parameter set to be used for training."
                         "Note that a hyper-parameter grid can be specified as using this argument"
                         "multiple times. In this case a grid-search will be performed and the best "
                         "hyper-parameter set will be chosen before the model is refitted (if selected)."
                         "See the models' source code for the meaning of the hyperparameters.")

    ap.add_argument('-m',
                    '--model',
                    action='store',
                    type=str,
                    default='bilstm2p',
                    help='The name of the neural network model architectire to be used.')

    return ap


def convert_arg_line_to_args(arg_line):
    """
    Custom line parser for argparse.
    :param arg_line: str
    One line of the input parameter file.
    :return: None
    """
    if arg_line:
        if arg_line[0] == '#':
            return
        for arg in arg_line.split():
            if not arg.strip():
                continue
            if '#' in arg:
                break
            yield arg
