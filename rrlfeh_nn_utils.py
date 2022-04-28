import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import sys
import functools
import os
import tensorflow.keras.backend as K
from matplotlib import pyplot as plt
# from IPython.display import clear_output
from scipy.stats import gaussian_kde, binned_statistic as binstat
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
from tensorflow.keras.losses import Loss
from scipy.spatial.distance import jensenshannon as js


class HuberLoss(Loss):
    """
    Custom TensorFlow Loss subclass implementing the Huber loss.
    """

    def __init__(self, threshold: float = 1):
        """
        :param threshold: float
        The Huber threshold between L1 and L2 losses.
        """
        super().__init__()
        self.threshold = threshold

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= self.threshold
        small_error_loss = tf.square(error) / 2
        big_error_loss = self.threshold * (tf.abs(error) - (0.5 * self.threshold))
        return tf.where(is_small_error, small_error_loss, big_error_loss)


def root_mean_squared_error(y, y_pred, sample_weight=None):
    """
    Compute the root mean squared error metric.
    """
    value = mean_squared_error(y, y_pred, sample_weight=sample_weight)
    return np.sqrt(value)


def process_input_parameters(pars, min_folds_cv=5):
    """
    Check the consistency of the input parameters and make modifications if necessary.
    :param pars: argparse.Namespace
        An argparse namespace object containing the input parameters.
    :param min_folds_cv: int
        The minimum number of folds required for K-fold cross-validation.
    :return: pars, argparse.Namespace
        The processed version of the input namespace object.
    """

    if len(pars.lcdir) > 1:
        assert len(pars.wavebands) == len(pars.lcdir), "The number of items in lcdir must either be 1 or match " \
                                                       "the number of items in wavebands."

    assert len(pars.wavebands) == len(pars.lcfile_suffices), \
        "The number of items in wavebands and lcfile_suffices must match."

    if not os.path.isdir(os.path.join(pars.rootdir, pars.outdir)):
        os.mkdir(os.path.join(pars.rootdir, pars.outdir))

    pars.hparam_grid = np.array(pars.hpars)

    # Check if only the CPU is to be used:
    if pars.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Join the list elements of pars.subset into a long string:
    if pars.subset:
        pars.subset = ' '.join(pars.subset)

    # Check the number of meta input features:
    if pars.meta_input is None:
        pars.n_meta = 0
    else:
        pars.n_meta = len(pars.meta_input)

    if pars.nn_type == 'cnn':
        pars.n_channels = len(pars.wavebands)
    else:
        pars.n_channels = 2 * len(pars.wavebands)

    if pars.weighing_by_density:
        print("Density weighing is ON with cutoff {}".format(pars.weighing_by_density))
    else:
        print("Density weighing is OFF.")
    print("Number of input channels: {}".format(pars.n_channels))
    print("Number of meta features: {}".format(pars.n_meta))

    if pars.train:

        pars.predict = False  # We want to train a regression model.

        if pars.pick_fold is not None:
            for ii in pars.pick_fold:
                print(type(ii))
                assert isinstance(ii, int) and 0 < ii <= pars.k_fold, \
                    "pick_fold must be > 0 AND <= k_fold integer"
            assert pars.k_fold >= min_folds_cv, \
                "pick_fold requires k_fold >= {}".format(min_folds_cv)
            pars.refit = False

        if not pars.cross_validate:
            assert len(pars.hparam_grid) == 1, "Cannot do grid-search of hyper-parameters if cross_validate is False."
            pars.refit = True

        if pars.explicit_test_frac:
            assert pars.refit or pars.ensemble, \
                "For the evaluation of the model on the test set, 'refit' or 'ensemble' must be set."

        if pars.optimize_lr:
            pars.n_epochs = 100
            pars.decay = 0.0
            pars.save_model = False
            pars.cross_validate = False
            pars.refit = True

    return pars


def read_dataset(filename: str, columns: list = None, subset_expr: str = None, input_feature_names: list = None,
                 trim_quantiles: list = None, qlo: float = 0.25, qhi: float = 0.75, plothist: bool = False,
                 histfig: str = "hist.png", dropna_cols: list = None, comment: str = '#', dtype=None):
    """
    Loads, trims, and exports dataset to numpy arrays.
    :param filename: str
    The name of the input file.

    :param columns: list of strings
    Passed to the usecols parameter of pandas.read_csv()

    :param subset_expr: str
    Expression for subsetting the input data, passed as the first parameter of pandas.DataFrame.query()

    :param input_feature_names: list of strings
    An optional subset of the usecols parameter, including the names of the columns to be returned as features.
    If None, all columns in usecols will be returned.

    :param trim_quantiles: list
    An optional subset of the usecols parameter, including the names of the columns to be threshold-rejected
    beyond the quantiles specified by qlo and qhi. If None, no quantile-trimming will be performed.

    :param qlo: float
    Lower quantile for threshold rejection.

    :param qhi: float
    Upper quantile for threshold rejection.

    :param plothist: bool
    If True, the histograms of the columns in usecols will be plotted before and, if performed, after quantile trimming.

    :param histfig: str
    The name of the output histogram figure file if plothist is True.

    :param dropna_cols:
    :param comment:
    :param dtype:
    :return:
    """

    with open(filename) as f:
        header = f.readline()
    cols = header.strip('#').split()
    df = pd.read_csv(filename, names=cols, header=None, sep="\s+", usecols=columns, comment=comment, dtype=dtype)
    if dropna_cols is not None:
        df.dropna(inplace=True, subset=dropna_cols)
    ndata = len(df)
    print(df.head())
    print("----------\n{} lines read from {}\n".format(ndata, filename))

    df_orig = df

    # Apply threshold rejections:
    if subset_expr is not None:
        df = df.query(subset_expr)

        ndata = len(df)
        print("{} lines after threshold rejections\n".format(ndata))

    # plot histogram for each column in original dataset
    if plothist:
        fig, ax = plt.subplots(figsize=(20, 10))
        fig.clf()
        _ = pd.DataFrame.hist(df, bins=int(np.ceil(np.cbrt(ndata) * 2)), figsize=(20, 10), grid=False, color='red',
                              ax=ax)
        plt.savefig(histfig)

    # omit data beyond specific quantiles [qlo, qhi]
    if trim_quantiles is not None:

        dfq = df[trim_quantiles]
        quantiles = pd.DataFrame.quantile(dfq, q=[qlo, qhi], axis=0, numeric_only=True, interpolation='linear')
        print("Values at [{},{}] quantiles to be applied for data trimming:".format(qlo, qhi))
        print(quantiles.sum)
        mask = (dfq > dfq.quantile(qlo)) & (dfq < dfq.quantile(qhi))
        # print(mask)
        mask = mask.all(axis=1)
        # print(mask.shape)
        df = pd.DataFrame.dropna(df[mask])
        ndata = len(df)
        print("\n{} lines remained after quantile rejection.\n".format(ndata))
        # plot histogram for each column in trimmed dataset
        if plothist:
            fig, ax = plt.subplots(figsize=(20, 10))
            _ = pd.DataFrame.hist(df, bins=int(np.ceil(np.cbrt(ndata) * 2)), figsize=(20, 10), grid=False,
                                  color='green', ax=ax)
            fig.savefig("hist_trim.png", format="png")

    if input_feature_names is not None:
        return df.loc[:, input_feature_names], df_orig
    else:
        return df, df_orig


def read_time_series_for_rnn(name_list, source_dir, nts, input_wavebands, ts_file_suffix, rootdir="",
                             periods=None, max_phase=1.0, phase_shift=None, nbins=None):

    print("Reading time series...", file=sys.stderr)
    n_data = len(name_list)
    scaler = StandardScaler(copy=True, with_mean=True, with_std=False)
    X_list = list()
    times_dict = dict()
    mags_dict = dict()
    phases_dict = dict()

    if nbins is not None:
        print("Light curves will be binned to max. {0} points in [0, {1:.1f}].".format(nbins, max_phase))

    for iband, waveband in enumerate(input_wavebands):

        X = np.zeros((n_data, nts, 2))  # Input shape required by an RNN: (batch_size, time_steps, features)

        phases = list()
        times = list()
        mags = list()

        if len(source_dir) > 1:
            directory = source_dir[iband]
        else:
            directory = source_dir[0]

        for ii, name in enumerate(name_list):

            print('Reading data for {}\r'.format(name), end="", file=sys.stderr)

            pp, mm = np.genfromtxt(os.path.join(rootdir, directory, name + ts_file_suffix[iband]),
                                   unpack=True, comments='#')
            phasemask = (pp < max_phase)
            pp = pp[phasemask]
            mm = mm[phasemask]

            if phase_shift is not None:
                pp = get_phases(1.0, pp, shift=phase_shift, all_positive=True)
                inds = np.argsort(pp)
                pp = pp[inds]
                mm = mm[inds]

            if nbins is not None:
                pp, mm = binlc(pp, mm, nbins=nbins, max_y=max_phase)

            if periods is not None:
                tt = pp * periods[ii]
            else:
                tt = pp

            # here we only subtract the mean:
            mm = scaler.fit_transform(mm.reshape(-1, 1)).flatten()
            times.append(tt)
            mags.append(mm)
            phases.append(pp)

        times_padded = pad_sequences(times, maxlen=nts, dtype='float64', padding='post', truncating='post', value=-1)
        mags_padded = pad_sequences(mags, maxlen=nts, dtype='float64', padding='post', truncating='post', value=-1)

        X[:, :, 0] = times_padded
        X[:, :, 1] = mags_padded

        X_list.append(X)

        times_dict[waveband] = times
        mags_dict[waveband] = mags
        phases_dict[waveband] = phases

    # Create final data matrix for the time series:
    X = np.concatenate(X_list, axis=2)

    print("")

    return X, times_dict, mags_dict, phases_dict


def read_time_series_for_cnn(name_list, source_dir, nts, input_wavebands, ts_file_suffix, nuse=1,
                             rootdir="", n_aug=None):
    nmags = int(nts / nuse)
    n_data = len(name_list)

    if n_aug is not None:
        assert isinstance(n_aug, int) and n_aug > 0, \
            "n_aug must be a positive integer"

    dict_x_ts = dict()
    for waveband in input_wavebands:
        dict_x_ts[waveband] = np.zeros((n_data, nmags))
        if n_aug is not None:
            dict_x_ts[waveband] = np.zeros((n_data * n_aug, nmags))
            groups = np.zeros((n_data * n_aug))
    dict_x_ts_scaled = dict()

    print("Reading time series...", file=sys.stderr)

    for ii, name in enumerate(name_list):

        print('Reading data for {}\r'.format(name), end="", file=sys.stderr)

        for iband, waveband in enumerate(input_wavebands):

            if len(source_dir) > 1:
                directory = source_dir[iband]
            else:
                directory = source_dir[0]

            if n_aug is None:
                phases, timeseries = np.genfromtxt(os.path.join(directory, name + ts_file_suffix[iband]),
                                                   unpack=True, comments='#')
                phases = phases[0:nts]
                timeseries = timeseries[0:nts]
                dict_x_ts[waveband][ii][:] = timeseries[nuse - 1::nuse]
                groups = None

            else:
                tsinput = np.genfromtxt(os.path.join(directory, name + ts_file_suffix[iband]),
                                        unpack=False, comments='#')
                # check if there are n_aug+1 columns in the data matrix
                assert tsinput.shape[1] == n_aug + 1, \
                    "data matrix in " + os.path.join(directory, name + ts_file_suffix[iband]) + " has wrong shape"
                phases = tsinput[0:nts, 0]
                for jj in range(n_aug):
                    timeseries = tsinput[0:nts, jj + 1]
                    dict_x_ts[waveband][jj + ii * n_aug][:] = timeseries[nuse - 1::nuse]
                    groups[jj + ii * n_aug] = ii

            phases = phases[nuse - 1::nuse]

    # Scale the time series to the [0,1] range
    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))

    ts_list = list()
    for ii, waveband in enumerate(input_wavebands):
        scaler.fit(dict_x_ts[waveband].T)
        dict_x_ts_scaled[waveband] = (scaler.transform(dict_x_ts[waveband].T)).T
        ts_list.append(np.expand_dims(dict_x_ts_scaled[waveband], axis=2))

    # Create final data matrix for the time series:
    X = np.concatenate(ts_list, axis=2)

    print("")

    return X, dict_x_ts, dict_x_ts_scaled, phases, groups


def cross_validate(model, folds: list, x_list: list or tuple, y,
                   model_kwargs: dict = {}, compile_kwargs: dict = {},
                   initial_weights: list = None,
                   sample_weight_fit=None, sample_weight_eval=None, ids=None,
                   indices_to_scale: list or tuple = None, scaler=None,
                   n_epochs: int = 1, batch_size: int = None, shuffle=True, verbose: int = 0,
                   callbacks: list = [], metrics: list or tuple = None,
                   log_training=True, log_prefix='', pick_fold: list or tuple = None,
                   save_data=True, rootdir='.', filename_train='train.dat', filename_val='val.dat',
                   strategy=None, n_devices=1, validation_freq=1, seed=1):

    # Initialize variables:
    histories = list()
    model_weights = list()
    scalers_folds = list()
    Y_train_collected = np.array([])
    Y_val_collected = np.array([])
    Y_train_pred_collected = np.array([])
    Y_val_pred_collected = np.array([])
    fitting_weights_train_collected = np.array([])
    fitting_weights_val_collected = np.array([])
    eval_weights_train_collected = np.array([])
    eval_weights_val_collected = np.array([])
    ids_train_collected = np.array([])
    ids_val_collected = np.array([])
    numcv_t = np.array([])
    numcv_v = np.array([])

    # callbacks.append(PrintLearningRate())

    if ids is None:
        # create IDs by simply numbering the data
        ids = np.linspace(1, y.shape[0], y.shape[0]).astype(int)

    first_fold = True

    for i_cv, (train_index, val_index) in enumerate(folds):

        # if pick_fold is not None and pick_fold != i_cv + 1:
        if pick_fold is not None and i_cv + 1 not in pick_fold:
            continue

        # Create and compile the model:
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        if strategy is not None:
            # Apply distributed strategy on model if multiple devices are present:
            with strategy.scope():
                model_ = model(**model_kwargs)
        else:
            model_ = model(**model_kwargs)

        if first_fold:
            first_fold = False
            model_.summary()

            if initial_weights is None:
                initial_weights = model_.get_weights()
        else:
            # Initialize model weights:
            model_.set_weights(initial_weights)

        if strategy is not None:
            with strategy.scope():
                model_.compile(**compile_kwargs)
        else:
            model_.compile(**compile_kwargs)

        print("fold " + str(i_cv + 1) + "/" + str(len(folds)))
        print("n_train = {}  ;  n_val = {}".format(train_index.shape[0], val_index.shape[0]))

        if log_training:
            callbacks_fold = callbacks + [tf.keras.callbacks.CSVLogger(
                os.path.join(rootdir, log_prefix + f"_fold{i_cv + 1}.log"))]
        else:
            callbacks_fold = callbacks

        # --------------------------------------------------
        # Split the arrays to training and validations sets:

        x_train_list = list()
        x_val_list = list()
        scalers = list()
        for i, x in enumerate(x_list):
            x_t, x_v = x[train_index], x[val_index]
            if indices_to_scale is not None and i in indices_to_scale:
                scaler.fit(x_t)
                x_t = scaler.transform(x_t)
                x_v = scaler.transform(x_v)
                scalers.append(scaler.copy())
            x_train_list.append(x_t)
            x_val_list.append(x_v)

        y_train, y_val = y[train_index], y[val_index]

        if sample_weight_fit is not None:
            fitting_weights_train, fitting_weights_val = sample_weight_fit[train_index], sample_weight_fit[val_index]
        else:
            fitting_weights_train, fitting_weights_val = None, None

        if sample_weight_eval is not None:
            eval_weights_train, eval_weights_val = sample_weight_eval[train_index], sample_weight_eval[val_index]
        else:
            eval_weights_train, eval_weights_val = None, None

        ids_t, ids_v = ids[train_index], ids[val_index]

        # --------------------------------------------------
        # Fit and evaluate the model for this fold:

        history = model_.fit(x=x_train_list, y=y_train, sample_weight=fitting_weights_train,
                             epochs=n_epochs, initial_epoch=0, batch_size=batch_size, shuffle=shuffle,
                             validation_data=(x_val_list, y_val, fitting_weights_val),
                             verbose=verbose, callbacks=callbacks_fold, validation_freq=validation_freq)
        Y_train_pred = (model_.predict(x_train_list)).flatten()
        Y_val_pred = (model_.predict(x_val_list)).flatten()
        histories.append(history)
        model_weights.append(model_.get_weights())
        scalers_folds.append(scalers.copy())

        # --------------------------------------------------
        # Append the values of this fold to those from the previous fold(s).

        Y_train_collected = np.hstack((Y_train_collected, y_train))
        Y_val_collected = np.hstack((Y_val_collected, y_val))
        Y_train_pred_collected = np.hstack((Y_train_pred_collected, Y_train_pred))
        Y_val_pred_collected = np.hstack((Y_val_pred_collected, Y_val_pred))

        if sample_weight_fit is not None:
            fitting_weights_train_collected = np.hstack((fitting_weights_train_collected, fitting_weights_train))
            fitting_weights_val_collected = np.hstack((fitting_weights_val_collected, fitting_weights_val))
        if sample_weight_eval is not None:
            eval_weights_train_collected = np.hstack((eval_weights_train_collected, eval_weights_train))
            eval_weights_val_collected = np.hstack((eval_weights_val_collected, eval_weights_val))
        if ids is not None:
            ids_train_collected = np.hstack((ids_train_collected, ids_t))
            ids_val_collected = np.hstack((ids_val_collected, ids_v))
        numcv_t = np.hstack((numcv_t, np.ones(Y_train_pred.shape).astype(int) * i_cv))
        numcv_v = np.hstack((numcv_v, np.ones(Y_val_pred.shape).astype(int) * i_cv))

        if save_data:
            val_arr = np.rec.fromarrays((ids_v, y_val, Y_val_pred),
                                        names=('id', 'true_val', 'pred_val'))
            train_arr = np.rec.fromarrays((ids_t, y_train, Y_train_pred),
                                          names=('id', 'true_train', 'pred_train'))
            np.savetxt(os.path.join(rootdir, filename_val + '_cv{}.dat'.format(i_cv + 1)), val_arr, fmt='%s %f %f')
            np.savetxt(os.path.join(rootdir, filename_train + '_cv{}.dat'.format(i_cv + 1)), train_arr, fmt='%s %f %f')

        # --------------------------------------------------
        # Compute and print the metrics for this fold:

        if metrics is not None:
            for metric in metrics:
                score_train = metric(y_train, Y_train_pred, sample_weight=eval_weights_train)
                score_val = metric(y_val, Y_val_pred, sample_weight=eval_weights_val)
                print(metric.__name__, "  (T) = {0:.3f}".format(score_train))
                print(metric.__name__, "  (V) = {0:.3f}".format(score_val))

    if save_data:
        val_arr = np.rec.fromarrays((ids_val_collected, numcv_v, Y_val_collected, Y_val_pred_collected),
                                    names=('id', 'fold', 'true_val', 'pred_val'))
        train_arr = np.rec.fromarrays((ids_train_collected, numcv_t, Y_train_collected, Y_train_pred_collected),
                                      names=('id', 'fold', 'true_train', 'pred_train'))
        np.savetxt(os.path.join(rootdir, filename_val + '.dat'), val_arr, fmt='%s %d %f %f')
        np.savetxt(os.path.join(rootdir, filename_train + '.dat'), train_arr, fmt='%s %d %f %f')

    cv_train_output = (Y_train_collected, Y_train_pred_collected,
                       eval_weights_train_collected, ids_train_collected, numcv_t)

    cv_val_output = (Y_val_collected, Y_val_pred_collected,
                     eval_weights_val_collected, ids_val_collected, numcv_v)

    return cv_train_output, cv_val_output, model_weights, scalers_folds, histories


class PrintLearningRate(tf.keras.callbacks.Callback):

    def __init__(self, print_freq=100):
        super().__init__()
        self.print_freq = print_freq
        self.i = 0

    def on_epoch_end(self, epoch, logs=None):
        if np.mod(self.i + 1, self.print_freq) == 0:
            print(f"epoch {self.i + 1}:")
            print(K.eval(self.model.optimizer.learning_rate))
            print(K.eval(self.model.optimizer.lr))

        self.i += 1


class PlotLearning(tf.keras.callbacks.Callback):
    """
    Custom Callback class for plotting the training progress on the fly.
    """

    def __init__(self, eval_metrics: list = None, n_zoom: int = 200, n_update: int = 20, figname: str = "liveplot"):
        """
        :param eval_metrics: list
            A list of performance evaluation metrics to be extracted from the training logs for plotting.
        :param n_zoom: int
            The loss and the metrics plots will be zoomed after n_zoom epochs have passed in order to make smaller
            improvements also visible after the initial rapid progress of the training.
        :param n_update: int
            The plot will be updated after every n_update traning epochs.
        :param figname: str
            The name of the output figure file (without extension.)
        """

        super().__init__()
        self.n_zoom = n_zoom
        self.n_update = n_update
        self.eval_metrics = eval_metrics
        self.n_metrics = len(eval_metrics)
        self.figname = figname
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.metr = [[] for _ in range(self.n_metrics)]
        self.val_metr = [[] for _ in range(self.n_metrics)]
        self.fig = None
        self.axes = None
        self.logs = []

    def on_train_begin(self, logs=None):
        self.fig, self.axes = \
            plt.subplots(2 + self.n_metrics, 1, sharex=False, figsize=(6, 4 + 2 * self.n_metrics))
        self.fig.subplots_adjust(bottom=0.06, top=0.98, hspace=0.15, left=0.07, right=0.8, wspace=0)

    def on_train_end(self, logs=None):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.metr = [[] for _ in range(self.n_metrics)]
        self.val_metr = [[] for _ in range(self.n_metrics)]
        self.fig.clf()
        plt.close(self.fig)
        plt.close('all')

    def on_epoch_end(self, epoch, logs=None):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        for ii in range(self.n_metrics):
            assert logs.__contains__(self.eval_metrics[ii]), "invalid metric: {}".format(self.eval_metrics[ii])
            self.metr[ii].append(logs.get(self.eval_metrics[ii]))
            self.val_metr[ii].append(logs.get('val_' + self.eval_metrics[ii]))
        self.i += 1

        if np.mod(self.i + 1, self.n_update) == 0:

            epochs = np.array(self.x) + 1

            self.axes[0].grid(True)
            self.axes[0].tick_params(axis='both', direction='in', labelleft=False, labelright=True)
            # clear_output(wait=True)
            self.axes[0].yaxis.tick_right()
            self.axes[0].plot(epochs, np.log10(self.losses), 'r-', label="TR", alpha=0.6)
            if self.val_losses[0] is not None:
                self.axes[0].plot(epochs, np.log10(self.val_losses), 'g-', label="CV", alpha=0.6)
                # print(np.min(self.val_losses))
            self.axes[0].set_ylabel('log(loss)')
            self.axes[0].legend(loc='upper left')

            self.axes[1].grid(True)
            self.axes[1].tick_params(axis='both', direction='in', labelleft=False, labelright=True)
            self.axes[1].yaxis.tick_right()
            log_tr_losses = np.log10(self.losses)
            self.axes[1].plot(epochs[-self.n_zoom:], log_tr_losses[-self.n_zoom:], 'r-', label="TR", alpha=0.6)
            if self.val_losses[0] is not None:
                log_val_losses = np.log10(self.val_losses)
                self.axes[1].plot(epochs[-self.n_zoom:], log_val_losses[-self.n_zoom:], 'g-', label="CV", alpha=0.6)
            if self.i > self.n_zoom:
                if self.val_losses[0] is not None:
                    minval = np.min([log_tr_losses[-self.n_zoom:].min(), log_val_losses[-self.n_zoom:].min()])
                    maxval = np.max([log_tr_losses[-self.n_zoom:].max(), log_val_losses[-self.n_zoom:].max()])
                else:
                    minval = log_tr_losses[-self.n_zoom:].min()
                    maxval = log_tr_losses[-self.n_zoom:].max()
                span = maxval - minval
                self.axes[1].set_ylim((minval - span / 10., maxval + span / 10.))
            self.axes[1].set_ylabel('log(loss)')

            for jj in range(self.n_metrics):

                self.axes[jj + 2].grid(True)
                self.axes[jj + 2].tick_params(axis='both', direction='in', labelleft=False, labelright=True)
                self.axes[jj + 2].yaxis.tick_right()
                self.axes[jj + 2].plot(epochs[-self.n_zoom:], self.metr[jj][-self.n_zoom:], 'r-', label="TR")
                if self.val_metr is not None:
                    self.axes[jj + 2].plot(epochs[-self.n_zoom:], self.val_metr[jj][-self.n_zoom:], 'g-', label="CV")
                if self.i > self.n_zoom:

                    if self.val_metr[jj][0] is not None:  # check if there is validation data for this metric
                        minval = np.min([np.array(self.metr[jj][-self.n_zoom:]).min(),
                                         np.array(self.val_metr[jj][-self.n_zoom:]).min()])
                        maxval = np.max([np.array(self.metr[jj][-self.n_zoom:]).max(),
                                         np.array(self.val_metr[jj][-self.n_zoom:]).max()])
                        span = maxval - minval
                        self.axes[jj + 2].set_ylim((minval - span / 10., maxval + span / 10.))
                    else:  # if there is no val. data for this metric, then compute extrema from training data only
                        minval = np.array(self.metr[jj][-self.n_zoom:]).min()
                        maxval = np.array(self.metr[jj][-self.n_zoom:]).max()
                        span = maxval - minval
                        self.axes[jj + 2].set_ylim((minval - span / 10., maxval + span / 10.))
                if jj == self.n_metrics - 1:
                    self.axes[jj + 2].set_xlabel('epoch')
                self.axes[jj + 2].set_ylabel(self.eval_metrics[jj])

            plt.savefig(self.figname + '.png', format='png')
            for ax in self.axes:
                ax.cla()


def reset_weights(model):
    """
    Reinitialize the weights of the model.
    Source: https://github.com/keras-team/keras/issues/341#issuecomment-547833394.
    :param model: a tensorflow.keras model instance.
    :return:
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):  # if you're using a model as a layer
            reset_weights(layer)  # apply function recursively
            continue

        # where are the initializers?
        if hasattr(layer, 'cell'):
            init_container = layer.cell
        else:
            init_container = layer

        for key, initializer in init_container.__dict__.items():
            if "initializer" not in key:  # is this item an initializer?
                continue  # if no, skip it

            # find the corresponding variable, like the kernel or the bias
            if key == 'recurrent_initializer':  # special case check
                var = getattr(init_container, 'recurrent_kernel')
            else:
                var = getattr(init_container, key.replace("_initializer", ""))
            if var is not None:
                var.assign(initializer(var.shape, var.dtype))
                # use the initializer


def setup_callbacks(auto_stop=None, min_delta=10e-5, patience=200,
                    optimize_lr=False, min_learning_rate=0.0001, n_training_epochs=100, lr_increment_coeff=0.9,
                    is_checkpoint=False, checkpoint_period=100,
                    save_model=False, n_zoom=100, n_update=100, eval_metrics=['accuracy'], figname="liveplot"):
    callbacks = []

    # Stop training automatically:
    if auto_stop is not None:
        if auto_stop == 'late':
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='loss', min_delta=min_delta, patience=patience, verbose=1, mode='min',
                baseline=None, restore_best_weights=False))
        elif auto_stop == 'early':  # i.e., early stopping
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', min_delta=min_delta, patience=patience, verbose=1, mode='min',
                baseline=None, restore_best_weights=True))

    # Change learning rate at each epoch to find optimal value:
    if optimize_lr:
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: min_learning_rate * 10 ** (epoch / (n_training_epochs * lr_increment_coeff)))
        callbacks.append(lr_schedule)

    if is_checkpoint and save_model:
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint('weights{epoch:03d}_{loss:.3f}.h5', monitor='loss',
                                                              save_best_only=False, save_weights_only=True,
                                                              period=checkpoint_period)
        callbacks.append(model_checkpoint)

    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=False, figsize=(6, 8))
    plotlearning = PlotLearning(n_zoom=n_zoom, n_update=n_update, eval_metrics=eval_metrics, figname=figname)
    callbacks.append(plotlearning)

    return callbacks


def plot_all_lc(phases, sequences, nmags=100, shift=0, fname=None, indx_highlight=None, figformat="png", nn_type='cnn'):
    """
    Plot all input time series as phase diagrams.
    """
    n_roll = int(np.round(shift * nmags))

    try:
        n_samples = sequences.shape[0]
    except:
        n_samples = len(sequences)

    if indx_highlight is not None:
        assert (indx_highlight >= 0)
        assert (indx_highlight == int(indx_highlight))
        assert (indx_highlight < n_samples)

    fig = plt.figure(figsize=(5, 4))
    fig.subplots_adjust(bottom=0.13, top=0.94, hspace=0.3, left=0.15, right=0.98, wspace=0)
    for ii in range(n_samples):
        if nn_type == "cnn":
            plt.plot(phases, np.roll(sequences[ii, :], n_roll), ls='-', color='grey', lw=0.3, alpha=0.3)
        elif nn_type == "rnn":
            ph = phases[ii]
            seq = sequences[ii]
            mask = (ph <= 1)
            seq = seq[mask]
            ph = ph[mask]
            # plt.plot(phases[ii], sequences[ii], ',', color='grey', alpha=0.5)
            plt.plot(ph, seq, ',', color='grey', alpha=0.5)
    if indx_highlight is not None:
        if nn_type == "cnn":
            plt.plot(phases, np.roll(sequences[indx_highlight, :], n_roll), 'ko')
        elif nn_type == "rnn":
            # plt.plot(phases[indx_highlight], sequences[indx_highlight], 'ko')
            ph = phases[indx_highlight]
            seq = sequences[indx_highlight]
            mask = (ph <= 1)
            seq = seq[mask]
            ph = ph[mask]
            plt.plot(ph, seq, 'ko')
    plt.xlabel('phase')
    plt.ylabel('mag')
    # plt.ylim(-1.1, 0.8)
    plt.gca().invert_yaxis()
    plt.savefig(fname + "." + figformat, format=figformat)
    plt.close(fig)


def plot_period_amplitude(parameter_matrix, col: int = 2, waveband: str = '', fname: str = None,
                          figformat: str = 'png'):
    """
    Plot the input amplitudes vs the input periods.
    """
    fig = plt.figure(figsize=(5, 4))
    plt.plot(np.log10(parameter_matrix[:, 0]), parameter_matrix[:, col], 'k.', alpha=0.1)
    plt.xlabel('log$P$ [d]')
    plt.ylabel('$A_{' + waveband + ',tot.}$ [mag]')
    if fname is not None:
        plt.savefig(fname + "." + figformat, format=figformat)
        plt.close(fig)
    else:
        plt.show()


def dev_test_split(*args, test_frac=0.1, groups=None, seed=None):
    """
    Split a list of matrices into development and test sets.

    A list of np.ndarray objects of the same shape. The same train-test split will be performed on all elements.

    :param args: np.ndarray or None
    Arrays on which to perform the train-test split.

    :param test_frac: float
    Ratio of the sizes of the test/train sets.

    :param groups: np.ndarray
    If not None, then this argument expects an array containing the group labels that will be passed to the
    scikit-learn GroupShuffleSplit splitter. If None, ShuffleSplit will be used instead.

    :param seed: int
    Optional random seed for reproducibility.

    :return dev_list, test_list: list, list
    The resulting lists of numpy.ndarray objects corresponding to the train / test sets.
    """

    # np.random.seed(seed=seed)  # random seed for data shuffling
    assert args, "at least one parameter must be passed."
    # check if we should do a train-test split:
    if test_frac is None:
        return args, None
    else:
        # check if test_frac makes sense:
        assert 0 <= test_frac < 1, "test_frac must be in [0,1)"
    # check if the elements of args are arrays and have the same shape:
    shapes = list()
    for arg in args:
        if arg is not None:
            assert isinstance(arg, np.ndarray), "elements of array_list must be np.ndarray or None"
            shapes.append(arg.shape)
    # print(shapes)
    assert all(shapes[0][0] == x[0] for x in shapes)
    n_data = shapes[0][0]

    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    else:
        splitter = ShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)

    for dvi, tsi in splitter.split(np.ones(n_data), groups=groups):
        dev_indx = dvi
        test_indx = tsi

    dev_list = []
    test_list = []
    for mm in args:
        if mm is not None:
            assert mm.shape[0] == n_data
            dims = len(mm.shape)
            if dims == 1:
                dev_list.append(mm[dev_indx])
                test_list.append(mm[test_indx])
            else:
                dev_list.append(mm[dev_indx, :])
                test_list.append(mm[test_indx, :])
        else:
            dev_list.append(None)
            test_list.append(None)

    return dev_list, test_list


def progress_plots(histories, filename, start_epoch: int = 0, title: str = None,
                   moving_avg: bool = False, beta: float = 0.9, plot_folds: bool = False):
    """
    Plot various metrics as a function of training epoch.

    :param histories: list
        List of history objects (each corresponding to a validation fold,
        captured from the output of the tf.keras.model.fit() method)
    :param filename: str
        Prefix for the name of the output figure file.
    :param start_epoch: int
        The first training epoch to be plotted.
    :param title: str
        Title of the figure.
    :param moving_avg: boolean
        If True, compute exponentially weighted moving averages (EWMA)
    :param beta: float
        Parameter for computing EWMA.
    :param plot_folds: boolean
        If True, individual cross-validation folds will be plotted in addition to their mean.
    """

    # Extract the list of metrics from histories:
    metrics = []
    for metric, _ in histories[0].history.items():
        if "val_" not in metric:
            metrics.append(metric)

    for metric in metrics:
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        fig.subplots_adjust(bottom=0.13, top=0.87, hspace=0.3, left=0.05, right=0.85, wspace=0)

        if title is not None:
            plt.title(title, fontsize=8)
        if metric == 'loss':
            plt.ylabel("log loss")
        else:
            plt.ylabel(metric)
        plt.xlabel('training epoch')
        train_seq = np.array([h.history[metric] for h in histories], dtype=object)

        if len(train_seq.shape) == 2:  # if there are the same number of epochs for each fold
            n_epochs = train_seq.shape[1]
            mean_train_seq = np.mean(train_seq, axis=0)
            epochs = np.linspace(1, n_epochs, n_epochs, endpoint=True).astype(int)

            # Plot training sequence:
            if metric == 'loss':
                train_seq = np.log10(train_seq)
                mean_train_seq = np.log10(mean_train_seq)
            line_train, = plt.plot(epochs[start_epoch:], mean_train_seq[start_epoch:], '-', c='darkred', linewidth=1,
                                   zorder=9, label='TR')
            if plot_folds:
                plt.plot(epochs[start_epoch:], train_seq.T[start_epoch:], 'r-', linewidth=1, zorder=8, alpha=0.3)

            if moving_avg:
                ewma_train = ewma(mean_train_seq, beta=beta)  # compute exponentially weighted moving averages
                plt.plot(epochs[start_epoch:], ewma_train[start_epoch:], 'k-', linewidth=1, zorder=10)

            if 'val_' + metric in histories[0].history:
                val_seq = np.array([h.history['val_' + metric] for h in histories])
                mean_val_seq = np.mean(val_seq, axis=0)
                if metric == 'loss':
                    val_seq = np.log10(val_seq)
                    mean_val_seq = np.log10(mean_val_seq)
                line_val, = plt.plot(epochs[start_epoch:], mean_val_seq[start_epoch:], 'g-', linewidth=1, zorder=11,
                                     label='CV')
                if plot_folds:
                    plt.plot(epochs[start_epoch:], val_seq.T[start_epoch:], 'g-', linewidth=1, zorder=12, alpha=0.3)
                if moving_avg:
                    ewma_val = ewma(mean_val_seq, beta)  # compute exponentially weighted moving averages
                    plt.plot(epochs[start_epoch:], ewma_val[start_epoch:], 'k-', linewidth=1, zorder=12)
                plt.legend(handles=[line_train, line_val], loc='upper left')
            else:
                plt.legend(handles=[line_train], loc='upper left')

        else:  # if each fold can have a different number of epochs

            # Plot training sequence:
            for tseq in train_seq:
                if metric == 'loss':
                    tseq = np.log10(np.array(tseq))
                else:
                    tseq = np.array(tseq)
                epochs = np.linspace(1, len(tseq), len(tseq), endpoint=True).astype(int)
                line_train, = plt.plot(epochs[start_epoch:], tseq[start_epoch:], 'r-', linewidth=1,
                                       zorder=8, alpha=0.5, label='TR')

            if 'val_' + metric in histories[0].history:
                val_seq = np.array([h.history['val_' + metric] for h in histories], dtype=object)
                for vseq in val_seq:
                    if metric == 'loss':
                        vseq = np.log10(np.array(vseq))
                    else:
                        vseq = np.array(vseq)
                    epochs = np.linspace(1, len(vseq), len(vseq), endpoint=True).astype(int)
                    line_val, = plt.plot(epochs[start_epoch:], vseq[start_epoch:], 'g-', linewidth=1,
                                         zorder=8, alpha=0.5, label='CV')
                    plt.legend(handles=[line_train, line_val], loc='upper left')
            else:
                plt.legend(handles=[line_train], loc='upper left')

        ax.tick_params(axis='both', direction='in', labelleft=False, labelright=True)
        ax.yaxis.tick_right()
        # if 'root_mean_squared_error' in metric:
        #     ax.set_ylim((0.1, 0.3))
        #     # plt.yticks(np.arange(0.0, 0.6, 0.1))
        # if 'loss' in metric:
        #     ax.set_ylim((-0.2, 0.25))
        # #     ax.set_yticks(np.arange(0.9, 1.005, 0.005))
        plt.grid(True)
        plt.savefig(filename + metric + '.pdf', format='pdf')
        plt.close(fig)


def ewma(y, beta=0.9, bias_corr=True):
    ma = np.zeros(y.shape)
    beta1 = 1.0 - beta

    v = 0
    for i in np.arange(len(y)) + 1:
        v = beta * v + beta1 * y[i - 1]
        if bias_corr:
            ma[i - 1] = v / (1. - beta ** i)
        else:
            ma[i - 1] = v
    return ma


def save_model(scaler_file="scaler", scaler=None, model_file="model", weights_file="model_weights",
               model=None, model_weights=None, suffix=""):
    # If either scaler or model is a sequence, then we are dealing with a model ensemble.
    if isinstance(scaler, (list, tuple, np.ndarray)) and isinstance(model_weights, (list, tuple, np.ndarray)):
        # In this case, check if scaler and model are both sequences and have the same length.
        assert len(scaler) == len(model_weights), "If saving an ensemble, " \
                                                  "both scaler and model must have the same length."

    if scaler is not None:
        if isinstance(scaler, (list, tuple, np.ndarray)):
            for ii, sc in enumerate(scaler):
                joblib.dump(scaler, scaler_file + '_' + str(ii) + '.sav')

        else:
            # Write out standard scaler to file:
            joblib.dump(scaler, scaler_file)

    # Serialize model to JSON:
    if model is not None:

        model_json = model.to_json()
        with open(model_file + suffix + ".json", "w") as json_file:
            json_file.write(model_json)

        if isinstance(model_weights, (list, tuple, np.ndarray)):
            for ii, ww in enumerate(model_weights):
                model.set_weights(ww)
                # Serialize weights to HDF5:
                model.save_weights(weights_file + suffix + '_' + str(ii) + ".h5")
        else:
            # Serialize weights to HDF5:
            model.save_weights(weights_file + suffix + ".h5")


def compute_regression_metrics(y, y_pred, metrics: dict = None, sample_weight=None):
    """
    Compute regression metrics and append them to list of metrics in a dictionary.
    :param y: numpy.ndarray
        Array of the true values.
    :param y_pred: numpy.ndarray
        Array of the predicted values.
    :param metrics: dict or None
        A dictionary of metric lists. If None, a new dictionary will be created.
    :param sample_weight: numpy.ndarray or None
        Array of the sample weights.
    :return: metrics: dict
        Dictionary of metric lists.
    """

    if metrics is not None:
        assert isinstance(metrics, dict), "metrics must be of type dict"
    else:
        metrics = {'r2': [], 'wrmse': [], 'wmae': [], 'rmse': [], 'mae': [], 'medae': []}
    assert isinstance(y, np.ndarray), "Y must be of type numpy.ndarray"
    assert isinstance(y_pred, np.ndarray), "Y_pred must be of type numpy.ndarray"
    if sample_weight is not None:
        assert isinstance(sample_weight, np.ndarray), "sample_weights must be of type numpy.ndarray"

    if sample_weight is not None:
        metrics['r2'].append(r2_score(y, y_pred, sample_weight=sample_weight))
        metrics['wrmse'].append(np.sqrt(mean_squared_error(y, y_pred, sample_weight=sample_weight)))
        metrics['wmae'].append(mean_absolute_error(y, y_pred, sample_weight=sample_weight))
    metrics['rmse'].append(np.sqrt(mean_squared_error(y, y_pred)))
    metrics['mae'].append(mean_absolute_error(y, y_pred))
    metrics['medae'].append(median_absolute_error(y, y_pred))

    return metrics


def print_regression_metrics(metrics_1, item, metrics_1_name="training", metrics_2=None, metrics_2_name="   CV   "):
    assert isinstance(metrics_1, dict), "metrics_1 must be of type dict"

    if metrics_2 is not None:

        assert isinstance(metrics_2, dict), "metrics_2 must be of type dict"

        print("  metric  |  {0:10s} |  {1:10s} |".format(metrics_1_name, metrics_2_name))
        print(" --------------------------------------")
        for key in metrics_1.keys():
            print(" {0:8s} | {1:10.4f}  | {2:10.4f}  |".format(key, metrics_1[key][item], metrics_2[key][item]))
        print(" --------------------------------------\n")

    else:

        print("  metric |  {}  |".format(metrics_1_name))
        print(" ----------------------")
        for key in metrics_1.keys():
            print(" {0:8s} | {1:10.4f}  |".format(key, metrics_1[key][item]))
        print(" ----------------------\n")


def compute_sample_weights(y, y_err=None, error_weighting="inverse_squared", by_variance=True, by_density=None,
                           scaled=True, plot=False, filename=None, xlabel="y",
                           figformat='png'):
    assert isinstance(y, np.ndarray), "y must be of type ndarray"
    assert error_weighting == "inverse" or error_weighting == "inverse_squared", "error_weighting must be \"inverse\"" \
                                                                                 "or \"inverse_squared\""

    if by_variance:
        assert isinstance(y_err, np.ndarray), "y_err must be of type ndarray"
        assert y.shape == y_err.shape, "y and y_err must have the same shape"
        # weights based on variance:
        if error_weighting == "inverse_squared":
            weights_var = 1.0 / y_err ** 2
        elif error_weighting == "inverse":
            weights_var = 1.0 / y_err
    else:
        weights_var = np.ones(y.shape)

    if by_density:
        # compute KDE of y values
        kde = gaussian_kde(y)
        ykde = kde(y)

        density_weighted_y = y[ykde > by_density]
        y_min_dens = np.min(density_weighted_y)
        y_max_dens = np.max(density_weighted_y)

        weights_dens = 1.0 / ykde
        if y_max_dens is not None:
            weights_dens[y > y_max_dens] = 1.0 / kde(y_max_dens)
        if y_min_dens is not None:
            weights_dens[y < y_min_dens] = 1.0 / kde(y_min_dens)
    else:
        weights_dens = np.ones(y.shape)

    weights = weights_dens * weights_var

    if scaled:
        weights = MinMaxScaler(feature_range=(0.01, 1)).fit_transform(weights.reshape(-1, 1)).flatten()

    if plot:
        sort_indx = np.argsort(y)

        fig, ax1 = plt.subplots(1, 1, figsize=(5, 4))
        fig.subplots_adjust(bottom=0.13, top=0.95, hspace=0, left=0.1, right=0.85, wspace=0)
        ax2 = ax1.twinx()
        if by_density:
            ax2.plot(y, weights_dens / weights_dens.max(), 'k.', alpha=1, label="density weights")
            ax1.plot(y[sort_indx], ykde[sort_indx], 'b-', label="KDE")
        ax2.plot(y, weights, 'r,', alpha=1, label="weights")
        ax1.hist(y, facecolor='red', alpha=0.4, bins='sqrt', density=True, label="hist.")
        ax1.set_xlabel(xlabel)
        ax1.set_xlabel(xlabel)
        if scaled:
            ax1.set_ylabel('norm. density')
        else:
            ax1.set_ylabel('density')
        ax2.set_ylabel('weights')
        ax1.tick_params(direction='in')
        ax2.tick_params(direction='in')
        # ax1.legend(loc='upper left', bbox_to_anchor=(0.0, 1.2))
        # ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 1.2))
        if filename is not None:
            plt.savefig(filename + '.' + figformat, format=figformat)
        else:
            plt.show()
        plt.close(fig)

    # return_list = [weights, weights_var, weights_dens]
    #
    # if by_variance:
    #     return_list.append(weights_var)
    # if by_density:
    #     return_list.append(weights_dens)

    return weights, weights_var, weights_dens


def plot_predictions(y_train_true, y_train_pred, y_val_true=None, y_val_pred=None,
                     colors=None, suffix: str = '', figformat: str = 'png',
                     bins: str = 'sqrt', rootdir='.'):
    """
    Plot training and CV predictions vs true values and their histograms.
    """

    ll = np.linspace(-3, 0.5, 100)
    if y_val_true is not None and y_val_pred is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 10))
    else:
        fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(5, 5))
        ax2 = None
        ax1.set_xlabel('[Fe/H]')
    fig.subplots_adjust(bottom=0.13, top=0.87, hspace=0, left=0.15, right=0.95, wspace=0)
    if colors is not None:
        ax1.scatter(y_train_true, y_train_pred, s=2, marker='.', alpha=0.2, c=colors)
    else:
        ax1.scatter(y_train_true, y_train_pred, s=2, marker='.', c='k', alpha=0.2)
    ax1.plot(ll, ll, 'r-')
    # ax1.set_xlabel('[Fe/H]')
    ax1.set_ylabel('[Fe/H] (pred., T)')
    ax1.set_xlim((-3.1, 0.1))
    ax1.set_ylim((-3.1, 0.1))
    ax1.tick_params(direction='in')
    if ax2 is not None:
        if colors is not None:
            ax2.scatter(y_val_true, y_val_pred, s=2, marker='.', alpha=0.2, c=colors)
        else:
            ax2.scatter(y_val_true, y_val_pred, s=2, marker='.', c='k', alpha=0.2)
        ax2.plot(ll, ll, 'r-')
        ax2.set_xlabel('[Fe/H]')
        ax2.set_ylabel('[Fe/H] (pred., V)')
        ax2.set_xlim((-3.1, 0.1))
        ax2.set_ylim((-3.1, 0.1))
        ax2.tick_params(direction='in')
    plt.savefig(os.path.join(rootdir, 'pred_vs_true_' + suffix + '.' + figformat), format=figformat)
    plt.close(fig)

    # ---------------------------------------------------------------
    # Plot histograms of training and CV predictions and real values.

    if y_val_true is not None and y_val_pred is not None:
        fig, (ax12, ax34) = plt.subplots(2, 2, figsize=(10, 7))
        ax1, ax2 = ax12
        ax3, ax4 = ax34
    else:
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(5, 10))
        ax2, ax4 = None, None

    fig.subplots_adjust(bottom=0.13, top=0.87, hspace=0.2, wspace=0.2, left=0.15, right=0.95)

    if ax2 is not None:
        values, bins, _ = ax2.hist(y_val_true, facecolor='red', alpha=0.2, bins=bins, density=True)
        ax2.hist(y_val_pred, facecolor='black', alpha=0.2, bins=bins, density=True)
        ax2.set_xlabel('[Fe/H]')
        ax2.set_ylabel('norm. log count (V)')
        ax2.set_xlim((-2.6, 0.1))
        ax2.set_yscale('log')
        ax2.tick_params(direction='in')

    ax1.hist(y_train_true, facecolor='red', alpha=0.2, bins=bins, density=True, label="true")
    ax1.hist(y_train_pred, facecolor='black', alpha=0.2, bins=bins, density=True, label="predicted")
    ax1.set_xlabel('[Fe/H]')
    # ax1.set_xlabel('[Fe/H]')
    ax1.set_ylabel('norm. log count (T)')
    ax1.set_xlim((-2.6, 0.1))
    ax1.set_yscale('log')
    if ax2 is not None:
        ax1.legend(loc="upper right", bbox_to_anchor=(1.4, 1.2), fancybox=True, shadow=True, ncol=2)
    else:
        ax1.legend(loc="upper right", bbox_to_anchor=(0.8, 1.2), fancybox=True, shadow=True, ncol=2)
    ax1.tick_params(direction='in')

    ax3.hist(y_train_true, facecolor='red', alpha=0.2, bins=bins, density=True)
    ax3.hist(y_train_pred, facecolor='black', alpha=0.2, bins=bins, density=True)
    ax3.set_xlabel('[Fe/H]')
    ax3.set_ylabel('norm. count (T)')
    ax3.set_xlim((-2.6, 0.1))
    ax3.tick_params(direction='in')

    if ax4 is not None:
        ax4.hist(y_val_true, facecolor='red', alpha=0.2, bins=bins, density=True)
        ax4.hist(y_val_pred, facecolor='black', alpha=0.2, bins=bins, density=True)
        ax4.set_xlabel('[Fe/H]')
        ax4.set_xlabel('[Fe/H]')
        ax4.set_ylabel('norm. count (V)')
        ax4.set_xlim((-2.6, 0.1))
        ax4.tick_params(direction='in')

    plt.savefig(os.path.join(rootdir, 'pred_true_hist_' + suffix + '.' + figformat), format=figformat)
    plt.close(fig)


def clf_performance(y, y_gt, ref_label=1, verbose=True):
    """
    Computes various performance statistics from the output of a binary classifier
    or from a multi-class classifier binarized with respect to ref_label.

    Input: y   : predicted integer labels
           y_gt: array of ground truth integer labels
           ref_label: reference label (default: 1)

    Output: performance: dictionary of performance statistics.
            confusion_matrix
    """

    performance = {}

    n_gt_positives = np.sum([y_gt == ref_label])  # test examples with true label ref_label
    n_gt_negatives = np.sum([y_gt != ref_label])  # test examples with true label not ref_label
    n_positives = np.sum([y == ref_label])  # test examples predicted to be ref_label
    n_negatives = np.sum([y != ref_label])  # test examples predicted to be not ref_label

    n_true_positives = np.sum(np.logical_and(y == ref_label, y_gt == ref_label))
    n_false_positives = np.sum(np.logical_and(y == ref_label, y_gt != ref_label))
    n_true_negatives = np.sum(np.logical_and(y != ref_label, y_gt != ref_label))
    n_false_negatives = np.sum(np.logical_and(y != ref_label, y_gt == ref_label))

    precision = np.nan
    if n_true_positives + n_false_positives != 0:
        precision = float(n_true_positives) / float(n_true_positives + n_false_positives)

    recall = np.nan
    if n_true_positives + n_false_negatives != 0:
        recall = float(n_true_positives) / float(n_true_positives + n_false_negatives)

    f1 = 2.0 * (precision * recall) / (precision + recall)

    accuracy = np.nan
    if (n_true_positives + n_true_negatives + n_false_positives + n_false_negatives) != 0:
        accuracy = float(n_true_positives + n_true_negatives) / float(
            n_true_positives + n_true_negatives + n_false_positives + n_false_negatives)

    fap = np.nan
    if (n_false_positives + n_true_negatives) != 0:
        fap = float(n_false_positives) / float(n_false_positives + n_true_negatives)

    performance['precision'] = precision
    performance['recall'] = recall
    performance['f1'] = f1
    performance['accuracy'] = accuracy
    performance['fap'] = fap

    # confusion_matrix = np.array([n_true_positives,n_false_positives,n_false_negatives,n_true_negatives]).reshape(-1,2)
    confusion_matrix = np.array([n_true_negatives, n_false_positives, n_false_negatives, n_true_positives]).reshape(-1,
                                                                                                                    2)

    if verbose:
        print("-------------------------")
        print("Ground truth:")
        print("  {0:d} positives".format(n_gt_positives))
        print("  {0:d} negatives".format(n_gt_negatives))
        print("Predictions:")
        print("  {0:d} predicted test positives".format(n_positives))
        print("  {0:d} predicted test negatives".format(n_negatives))
        print("Confusion matrix:")
        print(confusion_matrix)
        print("Performance metrics:")
        print("  precision = {0:.3f}".format(precision))
        print("  recall = {0:.3f}".format(recall))
        print("  F1 = {0:.3f}".format(f1))
        print("  accuracy = {0:.3f}".format(accuracy))
        print("  false alarm probability = {0:.3f}".format(fap))
        print("-------------------------")

    return performance, confusion_matrix


def clf_performance_summary(performance_t, performance_v,
                            confusion_matrix_t, confusion_matrix_v):
    """
    Displays a text summary of the performance metrics of the result.
    """

    print('Performance on training set:')
    print('    P       R       F1      A       FAP')
    print('    {0:.3f}   {1:.3f}   {2:.3f}   {3:.3f}   {4:.3f}' \
          .format(performance_t['precision'], performance_t['recall'], performance_t['f1'], \
                  performance_t['accuracy'], performance_t['fap']))
    print('    confusion matrix (t):')
    print(confusion_matrix_t)

    print('Performance on validation set:')
    print('    P       R       F1      A       FAP')
    print('    {0:.3f}   {1:.3f}   {2:.3f}   {3:.3f}   {4:.3f}' \
          .format(performance_v['precision'], performance_v['recall'], performance_v['f1'], \
                  performance_v['accuracy'], performance_v['fap']))
    print('    confusion matrix (v):')
    print(confusion_matrix_v)


def get_phases(period, x, epoch=0.0, shift=0.0, all_positive=True):
    """
    Compute the phases of a monoperiodic time series.

    :param period: float

    :param x: 1-dimensional ndarray.
    The array of the time values of which we want to compute the phases.

    :param epoch: float, optional (default=0.0)
    Time value corresponding to zero phase.

    :param shift: float, optional (default=0.0)
    Phase shift wrt epoch.

    :param all_positive: boolean, optional (default=True)
    If True, the computed phases will be positive definite.

    :return:
    phases: 1-dimensional ndarray
    The computed phases with indices matching x.
    """

    phases = np.modf((x - epoch + shift * period) / period)[0]

    if all_positive:
        phases = all_phases_positive(phases)
    return phases


def all_phases_positive(pha):
    """
    Converts an array of phases to be positive definite.

    :param pha : 1-dimensional ndarray
    The phases to be modified.

    :return:
    pha: 1-dimensional ndarray
    Positive definite version of pha.
    """

    while not (pha >= 0).all():  # make sure that all elements are >=0
        pha[pha < 0] = pha[pha < 0] + 1.0
    return pha


def get_lr_metric(optimizer):
    """
    Function for returning the learning rate in the keras model.fit history, as if it were a metric.
    :param optimizer: a tensorflow optimizer instances
    :return: float32, the learning rate
    """

    def lr(y_true, y_pred):
        # return optimizer._decayed_lr(tf.float32)    # I use ._decayed_lr method instead of .lr
        return optimizer.lr

    return lr


def plot_loss_vs_lr(history, figformat='png'):
    """
    Create a training loss vs lerning rate figure.
    :param history: tensorflow.python.keras.callbacks.History object
        The history from the tensorflow.keras.model.fit method.
    :param figformat: str
        The format of the produced figure.
    :return: None
    """
    print("OPTIMIZATION OF THE LEARNING RATE:")
    learning_rates = np.array(history.history['lr'])
    print('minimum trial learning rate: {}'.format(learning_rates[0]))
    print('maximum trial learning rate: {}'.format(learning_rates[-1]))
    fig = plt.figure(figsize=(5, 4))
    fig.subplots_adjust(bottom=0.13, top=0.94, hspace=0.3, left=0.15, right=0.98, wspace=0)
    plt.semilogx(learning_rates, history.history["loss"])
    plt.xlabel('learning rate')
    plt.ylabel('training loss')
    plt.savefig("loss_vs_lr." + figformat, format=figformat)
    plt.close(fig)


def plot_residual(y: np.ndarray, yhat: np.ndarray, xlabel: str = "", ylabel: str = "", highlight: np.ndarray = None,
                  plotrange: tuple = None, fname: str = None, format="png"):
    """
    Plot predicted vs ground truth values.

    :param y: 1-d ndarray
    Array with the ground truth values.

    :param yhat: 1-d ndarray
    Array with the predicted values.

    :param xlabel: string
    Label for the x axis.

    :param ylabel: string
    Label for the y axis.

    :param highlight: boolean ndarray
    Boolean array matching the shape of y. True values will be highlighted in the plot.

    :param plotrange: tuple
    Range to be applied to both the x and y axes of the plot.

    :param fname: string
    Name of the output figure file.

    :param format: string
    Format of the output figure file.
    """

    fig = plt.figure(figsize=(5, 4))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(y, yhat, s=2, marker='.', c='k', alpha=0.2)
    if highlight is not None:
        plt.scatter(y[highlight], yhat[highlight], s=0.5, marker='x', alpha=0.5)
    minval = np.min((y.min(), yhat.min()))
    maxval = np.max((y.max(), yhat.max()))
    ax = plt.gca()
    ax.set_aspect('equal')
    if plotrange is not None:
        plt.xlim(plotrange)
        plt.ylim(plotrange)
        grid = np.linspace(plotrange[0], plotrange[1], 100)
    else:
        grid = np.linspace(minval, maxval, 100)
    plt.plot(grid, grid, 'r-')
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname + '.' + format, format=format)
        plt.close(fig)
    else:
        plt.show()


def compute_probs(data, bins=None):
    if bins is None:
        # use the sqrt rule
        bins = int(np.sqrt(len(data)))
    hist, bins = np.histogram(data, bins)
    probs = hist / data.shape[0]
    return probs, bins


def compute_jsd(y_train, y_train_pred, y_val, y_val_pred, base=2):

    p_val, bins = compute_probs(y_val, bins=None)
    p_val_pred, _ = compute_probs(y_val_pred, bins=bins)
    p_train, _ = compute_probs(y_train, bins=bins)
    p_train_pred, _ = compute_probs(y_train_pred, bins=bins)

    jsd_train = js(p_train, p_train_pred, base=base)**2
    jsd_val = js(p_val, p_val_pred, base=base)**2

    return jsd_train, jsd_val


def compute_jsd_kde(y_train, y_train_pred, y_val, y_val_pred, limits=None, ngrid=100, base=2):

    if limits is not None:
        y_grid_min, y_grid_max = limits
    else:
        y_grid_min = np.min((y_train.min(), y_train_pred.min(), y_val.min(), y_val_pred.min()))
        y_grid_max = np.max((y_train.max(), y_train_pred.max(), y_val.max(), y_val_pred.max()))

    y_grid = np.linspace(y_grid_min, y_grid_max, ngrid)

    dens_train = gaussian_kde(y_train)(y_grid)
    dens_train_pred = gaussian_kde(y_train_pred)(y_grid)
    dens_val = gaussian_kde(y_val)(y_grid)
    dens_val_pred = gaussian_kde(y_val_pred)(y_grid)

    jsd_train = js(dens_train, dens_train_pred, base=base)**2
    jsd_val = js(dens_val, dens_val_pred, base=base)**2

    return jsd_train, jsd_val


def binlc(x, y, nbins=100, max_y=1.0):

    bin_edges = np.linspace(0.0, max_y, nbins + 1)
    y_binned = binstat(x, y, statistic='mean', bins=bin_edges).statistic
    x_binned = binstat(x, x, statistic='mean', bins=bin_edges).statistic
    nanmask = np.isnan(y_binned)
    x_binned = x_binned[~nanmask]
    y_binned = y_binned[~nanmask]

    return x_binned, y_binned
