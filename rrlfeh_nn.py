import configparser
import sys
import importlib
import os
import glob
import numpy as np
import joblib
import tensorflow as tf
import rrlfeh_nn_utils as ut
import rrlfeh_nn_models as mm
import rrlfeh_nn_io as io
from time import time
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.model_selection import RepeatedStratifiedKFold, GroupKFold
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler

importlib.reload(io)
importlib.reload(ut)
importlib.reload(mm)
tic = time()

# ======================================================================================================================
#                                              P R E A M B L E

# ------------------------------
#  WIRED-IN PARAMETERS:

fehcolname = 'FeH'  # column name for the metallicity in the input file
feherrcolname = 'FeH_e'  # column name for the metallicity error in the input file
# idcolumn = 'gaia_DR2_source_id'    # column name for the identifier in the input file
idcolumn = "id"    # column name for the identifier in the input file
figformat = "pdf"
figsuffix = ""
indx_highlight = 40  # index of the input time series to be highlighted on the plots
checkpoint_period = 200  # After how many epochs should model weights be saved?
# nmags = 100  # number of phase points in synthetic light-curves for CNN
nmags = 80
nuse = 3  # use every nuse-th phase as an input feature ( => floor(nmags/nuse) features) for CNN
# Print results for every hyper-parameter combo in CV?
info_gridcv = True
# Minimum number of folds for k_fold CV. If pars.k_fold < min_folds_cv, then stratified shuffle-split is used.
min_folds_cv = 5
min_learning_rate = 1e-3
lr_increment_coeff = 0.48  # at each epoch, min_learning_rate will be multiplied by 10 ** (epoch * lr_increment_coeff)
validation_freq = 1

# ------------------------------
#  COMMAND-LINE PARAMETERS:

# Read parameters from a file or from the command line:
parser = io.argparser()
# print(len(sys.argv))
if len(sys.argv) == 1:
    # use default name for the parameter file
    pars = parser.parse_args([io.default_parameter_file])
else:
    pars = parser.parse_args()

pars = ut.process_input_parameters(pars, min_folds_cv=min_folds_cv)
np.random.seed(seed=pars.seed)      # random seed for data shuffling

# ------------------------------------------
# SET UP DEVICE(S):

# Check the number of GPUs and set identical memory growth:
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # check if there are any GPUs
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print("Number of GPUs:", len(gpus), "physical,", len(logical_gpus), "logical")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

    n_gpus = len(gpus)
else:
    n_gpus = 0

# Set up strategy for multi / single GPU or CPU:

if n_gpus > 1:
    # Calculate batch size in case of multi-device parallel training using mirrored strategy:
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    n_devices = strategy.num_replicas_in_sync
    print("Number of devices: ", n_devices)
    print('Mirrored strategy set for GPUs: ')
    for gpu in gpus:
        print(gpu.name)
    batch_size = pars.batch_size_per_replica * strategy.num_replicas_in_sync

elif n_gpus == 1:
    strategy = tf.distribute.get_strategy()     # default strategy that works on CPU and single GPU
    n_devices = strategy.num_replicas_in_sync
    print("Number of devices: ", n_devices)
    print('Default strategy set for GPU: ', gpus[0].name)
    batch_size = pars.batch_size_per_replica

else:
    strategy = tf.distribute.get_strategy()     # default strategy that works on CPU and single GPU
    n_devices = strategy.num_replicas_in_sync
    print("Number of devices: ", n_devices)
    print('Default strategy set for CPU')
    batch_size = pars.batch_size_per_replica

print("Global batch size = {}".format(batch_size))
print("Batch size per replica = {}".format(pars.batch_size_per_replica))

# ------------------------------------------
# LOSS, METRICS, OPTIMIZATION, MODEL:

# Set the loss, val. metrics, and the optmization algorithm:
with strategy.scope():
    loss = tf.keras.losses.MeanSquaredError()
    # loss = ut.HuberLoss(threshold=0.2)
    # performance evaluation metric to report during training:
    optimizer = tf.keras.optimizers.Adam(learning_rate=pars.learning_rate, beta_1=0.9, beta_2=0.999,
                                         epsilon=1e-07, amsgrad=False)
    # metrics = [tf.keras.metrics.RootMeanSquaredError(), ut.get_lr_metric(optimizer)]
    metrics = [tf.keras.metrics.RootMeanSquaredError()]

model = mm.available_models[pars.model]

# model = mm.conv3_2c_f233s1p3s2_gmp_fc64
# model = mm.cnn3_fc_model
# model = mm.lstm2_fc1_model
# model = mm.lstm2_model
# model = mm.bilstm2rd_fc1_model
# model = mm.bilstm2p_model

# ======================================================================================================================
#                                         D A T A    I N P U T

# ------------------------------------------
# READ AND SUBSET THE INPUT METADATA:

input_table, _ = ut.read_dataset(os.path.join(pars.rootdir, pars.input_file), columns=pars.columns,
                                 subset_expr=pars.subset, input_feature_names=pars.features, plothist=True,
                                 histfig=os.path.join(pars.rootdir, pars.outdir, "gfeh_nn_input_table_hist.png"),
                                 dropna_cols=None, comment='#', dtype={idcolumn: str})
n_data = len(input_table)
if pars.nn_type == "cnn":
    nmags = int(nmags / nuse)
else:
    if pars.nbins is not None:
        nmags = pars.nbins

print("Number of input phase points: {}".format(nmags))

ids = input_table[idcolumn].to_numpy().astype(str)
np.savetxt(os.path.join(pars.rootdir, pars.outdir, 'used_ids.lst'), ids.T, fmt="%s")

# Create data matrix for input layer 2 from time-series metadata:
if pars.meta_input is not None:
    X_meta_list = [input_table[feature].to_numpy() for feature in pars.meta_input]
    X_input2 = np.vstack(X_meta_list).T
    print("Shape of the input metadata matrix: {}".format(X_input2.shape))
else:
    X_input2 = None

# ------------------------------------------
# READ (AND PLOT) THE INPUT TIME SERIES:

if pars.nn_type == "cnn":
    # Read time series for a convolutional neural network
    X_input1, X_ts, X_ts_scaled, phases, groups = \
        ut.read_time_series_for_cnn(ids, pars.lcdir, nmags, pars.wavebands, pars.lcfile_suffices,
                                    rootdir=pars.rootdir, nuse=nuse, n_aug=pars.n_aug)
    # Plot input data:
    if pars.plot_input_data:
        for ii, waveband in enumerate(pars.wavebands):
            print("Plotting input data...")

            ut.plot_all_lc(phases, X_ts[waveband], nmags=nmags, shift=0.0, indx_highlight=indx_highlight,
                           fname=os.path.join(pars.rootdir, pars.outdir, waveband + "_lc_all"),
                           figformat=figformat)
            ut.plot_all_lc(phases, X_ts_scaled[waveband], nmags=nmags, shift=0.0, indx_highlight=indx_highlight,
                           fname=os.path.join(pars.rootdir, pars.outdir, waveband + "_lc_all_scaled"),
                           figformat=figformat)
            if X_input2 is not None:
                ut.plot_period_amplitude(X_input2, col=1 + ii, waveband=waveband, figformat=figformat,
                                         fname=os.path.join(pars.rootdir, pars.outdir, "logpP-A_" + waveband))

else:
    # Read time series for a recurrent neural network
    pars.n_aug = None  # augmented version not yet implemented
    groups = None
    if pars.meta_input:
        periods_input = None
    else:
        periods_input = input_table['period'].to_numpy()
    X_input1, times, mags, phases = \
        ut.read_time_series_for_rnn(ids, pars.lcdir, nmags, pars.wavebands,
                                    pars.lcfile_suffices, rootdir=pars.rootdir,
                                    periods=periods_input, max_phase=pars.max_phase, phase_shift=None, nbins=pars.nbins)

    # Plot input data:
    if pars.plot_input_data:
        print("Plotting input data...")
        for ii, waveband in enumerate(pars.wavebands):
            ut.plot_all_lc(phases[waveband], mags[waveband], shift=0.0, indx_highlight=indx_highlight,
                           fname=os.path.join(pars.rootdir, waveband + "_lc_all_rnn"),
                           figformat=figformat, nn_type="rnn")

# ======================================================================================================================
#                                                T R A I N I N G

if pars.train:

    print("Training epochs = " + str(pars.n_epochs))
    print("Learning rate = " + str(pars.learning_rate))
    print("Learning rate decay = " + str(pars.decay))

    hparams_best = None

    # COMPUTE SAMPLE WEIGHTS:

    Y_input = input_table['FeH'].to_numpy()
    Y_e_input = input_table['FeH_e'].to_numpy()

    weights, weights_var, weights_dens = \
        ut.compute_sample_weights(Y_input, y_err=Y_e_input, by_variance=True, by_density=pars.weighing_by_density,
                                  scaled=True, plot=True,
                                  filename=os.path.join(pars.rootdir, pars.outdir,
                                                        "Y_density_weighting"),
                                  xlabel="$[Fe/H]_I$",
                                  figformat=figformat)

    # ------------------------------------------
    # DEFINE DEVELOPMENT (TRAINING + CV) AND TEST SETS:

    dev_list, test_list = ut.dev_test_split(X_input1, X_input2, Y_input, ids, weights, weights_var,
                                            test_frac=pars.explicit_test_frac, groups=groups, seed=pars.seed)
    X_dev1, X_dev2, Y_dev, ids_dev, weights_dev, weights_var_dev = dev_list
    n_dev = Y_dev.shape[0]

    print("n_dev = " + str(n_dev))
    if test_list is not None:
        n_test = test_list[0].shape[0]
        print("n_test  = " + str(n_test))
    else:
        print("No explicit test sample.")

    # ------------------------------------------
    # PREPARE ARRAY FOR STRATIFICATION IN Y:

    isort = np.argsort(Y_dev)  # Indices of sorted Y values
    yi = np.zeros(n_dev)
    yi[isort] = np.arange(n_dev)  # Compute Y value order
    yi = np.floor(yi / 20).astype(int)  # compute phase labels for RepeatedStratifiedKFold
    if np.min(np.bincount(yi.astype(int))) < pars.k_fold:  # If too few elements are with last label, ...
        yi[yi == np.max(yi)] = np.max(yi) - 1  # ... the then change that label to the one preceding it

    # ------------------------------------------
    # CHOOSE CROSS-VALIDATION METHOD:

    if pars.k_fold >= min_folds_cv:
        splitter = RepeatedStratifiedKFold(n_splits=pars.k_fold, n_repeats=pars.n_repeats, random_state=pars.seed)
    else:
        print('Cross-validation will be performed by stratified shuffle-split because k_fold < {}'.format(min_folds_cv))
        if pars.n_aug is not None:
            sys.exit('Cannot handle augmented data because scikit-learn.model_selection.StratifiedShuffleSplit'
                     'does not implement group labels. Please specify a k_fold >= {} in order to perform'
                     'repeated stratified k-fold CV instead.'.format(min_folds_cv))
        else:
            splitter = StratifiedShuffleSplit(n_splits=pars.k_fold, test_size=pars.split_frac, random_state=pars.seed)
            pars.n_repeats = 1

    # ------------------------------------------
    # SETUP CALLBACKS:

    callbacks = ut.setup_callbacks(auto_stop=pars.auto_stop, min_delta=pars.min_delta, patience=pars.patience,
                                   optimize_lr=pars.optimize_lr, min_learning_rate=min_learning_rate,
                                   n_training_epochs=pars.n_epochs,
                                   lr_increment_coeff=lr_increment_coeff,
                                   is_checkpoint=pars.save_checkpoints, checkpoint_period=checkpoint_period,
                                   save_model=pars.save_model, n_zoom=pars.n_zoom, n_update=pars.n_update,
                                   eval_metrics=['root_mean_squared_error'], figname=pars.liveplotname)

    # ------------------------------------------
    # PERFORM GRID-SEARCH BY CROSS-VALIDATION:

    metrics_t = {'r2': [], 'wrmse': [], 'wmae': [], 'rmse': [], 'mae': [], 'medae': [], 'jsd_hist': [], 'jsd_kde': []}
    metrics_v = {'r2': [], 'wrmse': [], 'wmae': [], 'rmse': [], 'mae': [], 'medae': [], 'jsd_hist': [], 'jsd_kde': []}

    if pars.cross_validate:

        perf_out = open(os.path.join(pars.rootdir, pars.outdir, 'performance.dat'), 'w')
        perf_out.write("# hparams r2 wrmse wmae rmse mae medae\n")

        model_weights_grid = list()
        scaler_grid = list()

        for ii, hparams_trial in enumerate(pars.hparam_grid):

            # This tag will be used for naming output files.
            run_tag = model.__name__ + "_w" + str(pars.weighing_by_density) + \
                      "__" + '_'.join(map(str, hparams_trial)) + "_" \
                                                                 "_lr" + str(pars.learning_rate) + "_Nb" + str(
                batch_size)
            # + "_Nep" + "{:.0e}".format(pars.n_epochs)

            results_dir = os.path.join(pars.rootdir, pars.outdir, run_tag)
            if not os.path.isdir(results_dir):
                os.mkdir(results_dir)

            if info_gridcv:
                print('==========================================================')
                print('       Current hyper-parameters: [' + ','.join(map(str, hparams_trial)) + ']')

            # Define scaler and CV splitter:
            # ------------------------------
            metascaler = StandardScaler(copy=True, with_mean=True, with_std=True)
            splitter.random_state = pars.seed
            folds = list(splitter.split(X_dev1, yi, groups=groups))

            # Prepare input data:
            # -------------------
            if pars.meta_input:
                X_dev = (X_dev1, X_dev2)
                indices_to_scale = (1,)
            else:
                X_dev = (X_dev1,)
                indices_to_scale = None

            # Perform cross-validation:
            # -------------------------

            model_kwargs = {'n_timesteps': nmags, 'n_channels': pars.n_channels,
                            'n_meta': pars.n_meta, 'hparams': hparams_trial}

            compile_kwargs = {'optimizer': optimizer, 'loss': loss, 'metrics': metrics}

            cv_train_out, cv_val_out, model_weights, scalers, histories = \
                ut.cross_validate(model, folds, X_dev, Y_dev,
                                  model_kwargs=model_kwargs, compile_kwargs=compile_kwargs,
                                  sample_weight_fit=weights_dev, sample_weight_eval=weights_var_dev,
                                  ids=ids_dev, indices_to_scale=indices_to_scale,
                                  scaler=metascaler, n_epochs=pars.n_epochs, batch_size=batch_size, callbacks=callbacks,
                                  metrics=(r2_score, ut.root_mean_squared_error), log_training=pars.log_training,
                                  log_prefix=run_tag, pick_fold=pars.pick_fold, save_data=True, rootdir=results_dir,
                                  filename_train=pars.predict_train_output + model.__name__ + str(hparams_trial),
                                  filename_val=pars.predict_val_output + model.__name__ + str(hparams_trial),
                                  strategy=strategy, n_devices=n_devices, validation_freq=validation_freq,
                                  seed=pars.seed)

            model_weights_grid.append(model_weights)
            scaler_grid.append(scalers)

            Y_train_collected, Y_train_pred_collected, eval_weights_train_collected, ids_train_collected, numcv_t = \
                cv_train_out
            Y_val_collected, Y_val_pred_collected, eval_weights_val_collected, ids_val_collected, numcv_v = cv_val_out

            # Compute regression scores on concatenated CV samples:
            # -----------------------------------------------------
            metrics_t = ut.compute_regression_metrics(Y_train_collected, Y_train_pred_collected, metrics_t,
                                                      sample_weight=eval_weights_train_collected)
            metrics_v = ut.compute_regression_metrics(Y_val_collected, Y_val_pred_collected, metrics_v,
                                                      sample_weight=eval_weights_val_collected)

            jsdh_t, jsdh_v = \
                ut.compute_jsd(Y_train_collected, Y_train_pred_collected, Y_val_collected, Y_val_pred_collected)
            metrics_t['jsd_hist'].append(jsdh_t)
            metrics_v['jsd_hist'].append(jsdh_v)
            jsdk_t, jsdk_v = \
                ut.compute_jsd_kde(Y_train_collected, Y_train_pred_collected, Y_val_collected, Y_val_pred_collected)
            metrics_t['jsd_kde'].append(jsdk_t)
            metrics_v['jsd_kde'].append(jsdk_v)

            # Print results on screen:
            # ------------------------
            if info_gridcv:
                print("\n-------------------------------------------")
                print("\nhparams({}) = {}".format(ii, hparams_trial))
                print("-------------------------------------------")
                print("Regression metrics (on concatenated folds):")
                ut.print_regression_metrics(metrics_t, ii, metrics_1_name="training", metrics_2=metrics_v,
                                            metrics_2_name="   CV   ")
                print("r2 / jsd_hist (CV) = {0:.1f}".format(metrics_v['r2'][ii] / metrics_v['jsd_hist'][ii]))
                print("r2 / jsd_kde  (CV) = {0:.1f}".format(metrics_v['r2'][ii] / metrics_v['jsd_kde'][ii]))
            # Plot the learning curves:
            # -------------------------
            ut.progress_plots(histories, os.path.join(results_dir, 'progress_'),
                              start_epoch=50, moving_avg=False, plot_folds=(pars.k_fold > 1),
                              title=str(pars.model) + ', hpars: ' + str(hparams_trial) + ',\n batch size: ' +
                                    str(batch_size) + ', lr: ' + str(pars.learning_rate))

            # Plot training and CV predictions vs true values and their histograms:
            # ---------------------------------------------------------------------
            if pars.plot_prediction:
                ut.plot_predictions(Y_train_collected, Y_train_pred_collected,
                                    y_val_true=Y_val_collected, y_val_pred=Y_val_pred_collected,
                                    rootdir=results_dir, suffix=run_tag + figsuffix, figformat=figformat)

            # Write performance metrics to file:
            # ----------------------------------
            perf_out.write("{0:s} {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:.4f} {6:.4f}\n"
                           .format(str(hparams_trial), metrics_v['r2'][ii], metrics_v['wrmse'][ii],
                                   metrics_v['wmae'][ii],
                                   metrics_v['rmse'][ii], metrics_v['mae'][ii], metrics_v['medae'][ii]))

            # -------------------------
            # SAVE MODEL ENSEMBLE:

            if pars.save_model and pars.ensemble:
                if not pars.meta_input:
                    scalers = None
                ut.save_model(scaler_file=os.path.join(results_dir, pars.metascaler_file), scaler=scalers,
                              model_file=os.path.join(results_dir, pars.model_file_prefix + run_tag),
                              weights_file=os.path.join(results_dir, pars.weights_file_prefix + run_tag),
                              model=model(**model_kwargs), model_weights=model_weights)

        perf_out.close()

        # ------------------------------------------
        # DETERMINE THE BEST SET OF HYPERPARAMETERS:

        performances = np.array(metrics_v[pars.eval_metric])
        ibest = np.unravel_index(performances.argmax(), performances.shape)[0].astype(int)
        hparams_best = pars.hparam_grid[ibest]
        model_weights_best = model_weights_grid[ibest]
        model_best = model(n_timesteps=nmags, n_channels=pars.n_channels,
                           n_meta=pars.n_meta, hparams=hparams_best)
        scalers_best = scaler_grid[ibest]

        if len(pars.hparam_grid) > 1:
            print('==========================================================')
            print('--------------------     RESULTS     ---------------------')
            print("\n-------------------------------------------")
            print("optimal hparams = {}".format(hparams_best))
            print("optimization metric: {}".format(pars.eval_metric))
            print("-------------------------------------------")
            print("Regression metrics (on concatenated folds):")
            ut.print_regression_metrics(metrics_t, ibest, metrics_1_name="training", metrics_2=metrics_v,
                                        metrics_2_name="   CV   ")

        # --------------------------------------------------
        # EVALUATE BEST MODEL ENSEMBLE ON EXPLICIT TEST SET:

        if pars.explicit_test_frac and pars.ensemble:

            X_test1, X_test2, Y_test, ids_test, weights_test, weights_var_test = test_list
            Y_test_pred_list = list()
            for weights, scaler in zip(model_weights_best, scalers_best):
                model_best.set_weights(weights)
                if pars.meta_input:
                    X_test = (X_test1, scaler.transform(X_test2))
                else:
                    X_test = (X_test1,)
                Y_test_pred_list.append(model_best.predict(X_test).flatten())
            Y_test_pred = np.mean(np.vstack(Y_test_pred_list), axis=0)
            metrics_test = ut.compute_regression_metrics(Y_test, Y_test_pred, sample_weight=weights_test)
            print("----------------------------------------")
            print("Performance on explicit test set:\n")
            ut.print_regression_metrics(metrics_test, 0, metrics_1_name="  TEST  ")

            test_arr = np.rec.fromarrays((ids_test, Y_test, Y_test_pred),
                                         names=('id', 'true_test', 'pred_test'))
            np.savetxt(pars.predict_test_output + ".dat", test_arr, fmt='%s %f %f')

    # ------------------------------------------
    # REFIT THE BEST MODEL ON THE ENTIRE DEVELOPMENT SET:

    if pars.refit:
        print('------------------------------------')
        print("Fitting model to the development set...")

        if hparams_best is None:
            hparams_best = pars.hparam_grid[0]

        run_tag = model.__name__ + "_w" + str(pars.weighing_by_density) + "_" + str(hparams_best) + \
                  "_lr" + str(pars.learning_rate) + "_Nb" + str(batch_size) + "_Nep" + "{:.0e}".format(pars.n_epochs)

        if pars.log_training:
            callbacks.append(tf.keras.callbacks.CSVLogger(run_tag + ".log"))

        tf.keras.backend.clear_session()
        tf.random.set_seed(pars.seed)
        # Define and compile model:

        with strategy.scope():
            model_ = model(n_timesteps=nmags, n_channels=pars.n_channels, n_meta=pars.n_meta, hparams=hparams_best)
            model_.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        if pars.meta_input:
            # Standard-scale data:
            metascaler = StandardScaler(copy=True, with_mean=True, with_std=True)
            metascaler.fit(X_dev2)
            X_dev2 = metascaler.transform(X_dev2)
            X_dev = (X_dev1, X_dev2)
        else:
            metascaler = None
            X_dev = (X_dev1,)

        hist_refit = model_.fit(x=X_dev, y=Y_dev, epochs=pars.n_epochs, batch_size=batch_size,
                                sample_weight=weights_dev, initial_epoch=0, verbose=0,
                                callbacks=callbacks, shuffle=True)
        Y_dev_pred = (model_.predict(X_dev)).flatten()

        if pars.optimize_lr:
            ut.plot_loss_vs_lr(hist_refit, figformat=figformat)

        metrics_dev = ut.compute_regression_metrics(Y_dev, Y_dev_pred, sample_weight=weights_dev)

        print("----------------------------------------")
        print("Performance on refitted development set:\n")
        ut.print_regression_metrics(metrics_dev, 0, metrics_1_name="  DEV   ")

        if pars.save_model:
            ut.save_model(scaler_file=os.path.join(results_dir, pars.metascaler_file), scaler=metascaler,
                          model_file=os.path.join(results_dir, pars.model_file_prefix + run_tag),
                          weights_file=os.path.join(results_dir, pars.weights_file_prefix + run_tag), model=model_)
            print("\n -- Saved refitted model to disk. --")

        # Plot the loss and the accuracy as functions of the training epoch:
        ut.progress_plots([hist_refit], os.path.join(results_dir, pars.progress_plot_subdir, "refit_" + run_tag),
                          title=str(model_.name) + ', hpars: ' + str(hparams_best) + ',\n batch size: ' +
                                str(batch_size) + ', lr: ' + str(pars.learning_rate),
                          start_epoch=200, moving_avg=False, plot_folds=False)

        # ------------------------------------------
        # EVALUATE THE REFITTED MODEL ON AN EXPLICIT TEST SET:

        if pars.explicit_test_frac:

            X_test1, X_test2, Y_test, ids_test, weights_test, weights_var_test = test_list
            if pars.meta_input:
                X_test = (X_test1, metascaler.transform(X_test2))
            else:
                X_test = (X_test1,)
            Y_test_pred = model_.predict(X_test).flatten()
            metrics_test = ut.compute_regression_metrics(Y_test, Y_test_pred, sample_weight=weights_test)
            print("----------------------------------------")
            print("Performance on explicit test set:\n")
            ut.print_regression_metrics(metrics_test, 0, metrics_1_name="  TEST  ")

            test_arr = np.rec.fromarrays((ids_test, Y_test, Y_test_pred),
                                         names=('id', 'true_test', 'pred_test'))
            np.savetxt(pars.predict_test_output + ".dat", test_arr, fmt='%s %f %f')

# ======================================================================================================================
#                                                P R E D I C T I O N

if pars.predict:

    model_in = model(n_timesteps=nmags, n_channels=pars.n_channels, n_meta=pars.n_meta, hparams=pars.hpars[0])

    scalers_files_list = None

    if pars.ensemble:
        weights_files_list = glob.glob(os.path.join(pars.rootdir, pars.input_model_dir,
                                                    pars.weights_file_prefix + "_*.h5"))
        if pars.meta_input:
            scalers_files_list = [os.path.join(pars.rootdir, pars.input_model_dir, pars.metascaler_file + "_*.sav")]
    else:
        weights_files_list = [os.path.join(pars.rootdir, pars.input_model_dir, pars.weights_file_prefix + ".h5")]
        if pars.meta_input:
            scalers_files_list = [os.path.join(pars.rootdir, pars.input_model_dir, pars.metascaler_file + ".sav")]

    Y_pred_list = list()
    for ii, wf in enumerate(weights_files_list):

        model_in.load_weights(wf)

        if scalers_files_list:
            metascaler_in = joblib.load(scalers_files_list[ii])
            X_input = (X_input1, metascaler_in.transform(X_input2))
        else:
            X_input = (X_input1,)
        Y_pred_list.append(model_in.predict(X_input).flatten())

    if len(Y_pred_list) > 1:
        Y_pred = np.mean(np.vstack(Y_pred_list), axis=0)
        Y_pred_std = np.std(np.vstack(Y_pred_list), axis=0)
        outarr = np.rec.fromarrays((ids, Y_pred, Y_pred_std), names=('id', 'pred', 'pred_std'))
        fmt = '%s %f %f'
    else:
        outarr = np.rec.fromarrays((ids, Y_pred_list[0]), names=('id', 'pred'))
        fmt = '%s %f'
    np.savetxt(os.path.join(pars.rootdir, pars.target_output_file), outarr, fmt=fmt)

# ======================================================================================================================
# Create timestamp:
toc = time()
print("\n--- Execution time: {0:.1f} seconds ---".format(toc - tic))
