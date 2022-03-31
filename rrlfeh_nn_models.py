from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding1D, BatchNormalization, Flatten, Conv1D
from tensorflow.keras.layers import Dropout, GlobalMaxPooling1D,  \
    MaxPooling1D, concatenate, LeakyReLU, Bidirectional, Masking, LSTM, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.initializers import glorot_uniform
from itertools import cycle


def cnn3_fc_model(n_phases=45, n_channels=1, n_meta=3, regpars=(0, 0.12, 0.14, 0.5)):
    regpar_queue = cycle(regpars)

    x_input = Input(shape=(n_phases, n_channels), name='mags_input')
    xm_input = Input(shape=(n_meta,), name='meta_input')

    x = conv1d_block(x_input, n_filters=16, kernel_size=2, conv_stride=1, dropout=next(regpar_queue))  # 16

    x = conv1d_block(x, n_filters=32, kernel_size=3, conv_stride=1, dropout=next(regpar_queue))  # 48

    x = conv1d_block(x, n_filters=48, kernel_size=3, conv_stride=1, dropout=next(regpar_queue))  # 64

    x = Flatten()(x)
    x = concatenate([x, xm_input])

    x = Dense(64, activation=None)(x)
    x = LeakyReLU()(x)
    x = Dropout(rate=next(regpar_queue))(x)

    x = Dense(1, activation=None, name='head')(x)

    model = Model(inputs=[x_input, xm_input], outputs=x, name='cnn3_fc_model')
    return model


def cnn3_gmp_fc_model(n_phases=45, n_channels=1, n_meta=3, regpars=(0, 0.12, 0.14, 0.5)):
    regpar_queue = cycle(regpars)

    x_input = Input(shape=(n_phases, n_channels), name='mags_input')
    xm_input = Input(shape=(n_meta,), name='meta_input')

    x = conv1d_block(x_input, n_filters=16, kernel_size=2, conv_stride=1, dropout=next(regpar_queue))

    x = conv1d_block(x, n_filters=48, kernel_size=3, conv_stride=1, dropout=next(regpar_queue))

    x = conv1d_block(x, n_filters=64, kernel_size=3, conv_stride=1, pooling=False, dropout=None)

    x = GlobalMaxPooling1D()(x)
    x = Dropout(rate=next(regpar_queue))(x)

    x = concatenate([x, xm_input])

    x = Dense(64, activation=None)(x)
    x = LeakyReLU()(x)
    x = Dropout(rate=next(regpar_queue))(x)

    x = Dense(1, activation=None, name='head')(x)

    model = Model(inputs=[x_input, xm_input], outputs=x, name='cnn3_gmp_fc_model')
    return model


def conv1d_block(x, n_filters=16, kernel_size=3, conv_stride=1, conv_padding='same',
                 batch_norm=True, pooling=True, pool_size=3, pool_stride=2, pool_padding='valid',
                 dropout=None):
    x = Conv1D(n_filters, kernel_size, strides=conv_stride, padding=conv_padding, activation=None, use_bias=True,
               kernel_initializer=glorot_uniform(seed=0),
               kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1)(x)
    # x = Activation('elu')(x)
    x = LeakyReLU()(x)

    if pooling:
        x = MaxPooling1D(pool_size=pool_size, strides=pool_stride, padding=pool_padding)(x)

    if dropout is not None:
        x = Dropout(rate=dropout)(x)

    return x


def bilstm2_fc1_model(n_phases=100, regpars=(1e-6, 0.0, 0.2, 0.3, 0.5), n_channels=2, n_meta=2):

    l1reg1, l1reg2, dropout_lstm1, dropout_lstm2, dropout_fc1 = regpars

    xs_input = Input(shape=(None, n_channels), name='seq_input')
    xm_input = Input(shape=(n_meta,), name='meta_input')

    mask = Masking(mask_value=-999, input_shape=(n_phases, n_channels)).compute_mask(xs_input)

    # x = Dropout(rate=dprob_input)(xs_input)

    # x = Masking(mask_value=-999, input_shape=(100, 2))(xm_input)
    x = Bidirectional(LSTM(30, return_sequences=True,
                           recurrent_regularizer=L1L2(l1=l1reg1, l2=0)))(xs_input, mask=mask)
    x = Dropout(rate=dropout_lstm1)(x)
    x = Bidirectional(LSTM(30, return_sequences=False,
                           recurrent_regularizer=L1L2(l1=l1reg2, l2=0)))(x, mask=mask)
    x = Dropout(rate=dropout_lstm2)(x)

    x = concatenate([x, xm_input])

    # x = Dropout(rate=dprob)(x)

    x = Dense(64, activation='relu', name='fc1')(x)
    x = Dropout(rate=dropout_fc1)(x)

    x = Dense(1, activation=None, name='head')(x)

    model = Model(inputs=[xs_input, xm_input], outputs=x, name='bilstm2_fc1')
    return model


def bilstm2p_model(n_timesteps=100, n_channels=2, mask_value=-1,
                   hparams=(16, 16, 'l2', 2e-6, 2e-6, 2e-6, 2e-6, 0.2, 0.2), n_meta=None):

    dim1, dim2, rtype, rec_r1, rec_r2, kernel_r1, kernel_r2, dropout1, dropout2 = hparams

    assert rtype == 'l1' or rtype == 'l2', "rtype must be either 'l1' or 'l2'"

    if rtype == 'l1':
        kernel_regularizer1 = L1L2(l1=float(kernel_r1), l2=0)
        kernel_regularizer2 = L1L2(l1=float(kernel_r2), l2=0)
        recurrent_regularizer1 = L1L2(l1=float(rec_r1), l2=0)
        recurrent_regularizer2 = L1L2(l1=float(rec_r2), l2=0)
    else:
        kernel_regularizer1 = L1L2(l1=0, l2=float(kernel_r1))
        kernel_regularizer2 = L1L2(l1=0, l2=float(kernel_r2))
        recurrent_regularizer1 = L1L2(l1=0, l2=float(rec_r1))
        recurrent_regularizer2 = L1L2(l1=0, l2=float(rec_r2))

    xs_input = Input(shape=(None, n_channels), name='seq_input')

    mask = Masking(mask_value=mask_value, input_shape=(n_timesteps, n_channels)).compute_mask(xs_input)

    x = Bidirectional(LSTM(int(dim1), return_sequences=True, kernel_regularizer=kernel_regularizer1,
                           recurrent_regularizer=recurrent_regularizer1))(xs_input, mask=mask)

    if float(dropout1) > 0:
        x = Dropout(rate=float(dropout1))(x)

    x = Bidirectional(LSTM(int(dim2), return_sequences=False, kernel_regularizer=kernel_regularizer2,
                           recurrent_regularizer=recurrent_regularizer2))(x)

    if float(dropout2) > 0:
        x = Dropout(rate=float(dropout2))(x)

    x = Dense(1, activation=None, name='head')(x)

    model = Model(inputs=xs_input, outputs=x, name='bilstm2p')
    return model


def bilstm3p_model(n_timesteps=100, n_channels=2, mask_value=-1,
                   hparams=(16, 16, 16, 'l2', 2e-6, 2e-6, 2e-6, 2e-6, 2e-6, 2e-6, 0.2, 0.2, 0.2), n_meta=None):

    dim1, dim2, dim3, rtype, rec_r1, rec_r2, rec_r3, kernel_r1, kernel_r2, kernel_r3, dropout1, dropout2, dropout3 = \
        hparams

    assert rtype == 'l1' or rtype == 'l2', "rtype must be either 'l1' or 'l2'"

    if rtype == 'l1':
        kernel_regularizer1 = L1L2(l1=float(kernel_r1), l2=0)
        kernel_regularizer2 = L1L2(l1=float(kernel_r2), l2=0)
        kernel_regularizer3 = L1L2(l1=float(kernel_r3), l2=0)
        recurrent_regularizer1 = L1L2(l1=float(rec_r1), l2=0)
        recurrent_regularizer2 = L1L2(l1=float(rec_r2), l2=0)
        recurrent_regularizer3 = L1L2(l1=float(rec_r3), l2=0)
    else:
        kernel_regularizer1 = L1L2(l1=0, l2=float(kernel_r1))
        kernel_regularizer2 = L1L2(l1=0, l2=float(kernel_r2))
        kernel_regularizer3 = L1L2(l1=0, l2=float(kernel_r3))
        recurrent_regularizer1 = L1L2(l1=0, l2=float(rec_r1))
        recurrent_regularizer2 = L1L2(l1=0, l2=float(rec_r2))
        recurrent_regularizer3 = L1L2(l1=0, l2=float(rec_r3))

    xs_input = Input(shape=(None, n_channels), name='seq_input')

    mask = Masking(mask_value=mask_value, input_shape=(n_timesteps, n_channels)).compute_mask(xs_input)

    x = Bidirectional(LSTM(int(dim1), return_sequences=True, kernel_regularizer=kernel_regularizer1,
                           recurrent_regularizer=recurrent_regularizer1))(xs_input, mask=mask)

    if float(dropout1) > 0:
        x = Dropout(rate=float(dropout1))(x)

    x = Bidirectional(LSTM(int(dim2), return_sequences=True, kernel_regularizer=kernel_regularizer2,
                           recurrent_regularizer=recurrent_regularizer2))(x)

    if float(dropout2) > 0:
        x = Dropout(rate=float(dropout2))(x)

    x = Bidirectional(LSTM(int(dim3), return_sequences=False, kernel_regularizer=kernel_regularizer3,
                           recurrent_regularizer=recurrent_regularizer3))(x)

    if float(dropout3) > 0:
        x = Dropout(rate=float(dropout3))(x)

    x = Dense(1, activation=None, name='head')(x)

    model = Model(inputs=xs_input, outputs=x, name='bilstm3p')
    return model


available_models = {'bilstm2p': bilstm2p_model, 'bilstm3p': bilstm3p_model,
                    'cnn3_fc': cnn3_fc_model, 'cnn3_gmp_fc_model': cnn3_gmp_fc_model}
