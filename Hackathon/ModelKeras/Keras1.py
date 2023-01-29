import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
import tensorflow as tf
import gc
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier


def hyper_grid_search():
    model = KerasClassifier(model=create_model, verbose=1)

    samplers = [RandomUnderSampler(sampling_strategy='majority'), RandomOverSampler(sampling_strategy='all')]
    df, t2d_X_train, t2d_X_test, t2d_y_train, t2d_y_test, ibd_X_train, ibd_X_test, ibd_y_train, ibd_y_test, cad_X_train, cad_X_test, cad_y_train, cad_y_test, ckd_X_train, ckd_X_test, ckd_y_train, ckd_y_test, n_features, label_count  = load_data(sampler=samplers[0])
    label_count = [label_count]
    n_features = [n_features]
    batch_sizes = [32, 64]
    epochss = [16, 32, 64]
    activations = ['relu']
    # activations = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    init_mode = ['he_normal']
    # init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    dropout_rates = [0.0, 0.1]
    kernel_regularizer_val1s = [1e-6]
    kernel_regularizer_val2s = [1e-5]
    bias_regularizer_vals = [1e-5]
    activity_regularizer_vals = [1e-6]
    opt_types = [1, 2]
    learning_rates = [0.0000001] # [0.0000001, 0.000001, 0.00001, 0.0001]
    dropouts = [True]
    batchNs = [True, False]
    sizes1 = [4] # [4, 8, 16, 32]
    sizes2 = [4]
    sizes3 = [4]
    param_grid = dict(model__activation=activations, model__dropout_rate=dropout_rates, model__init_mode=init_mode,
                      model__kernel_regularizer_val1=kernel_regularizer_val1s, model__kernel_regularizer_val2=kernel_regularizer_val2s,
                      model__bias_regularizer_val=bias_regularizer_vals, model__activity_regularizer_val=activity_regularizer_vals,
                      model__label_count=label_count, model__n_features=n_features,
                      model__opt_type=opt_types, model__learning_rate=learning_rates, model__dropout=dropouts, model__batchN=batchNs,
                      model__size1=sizes1, model__size2=sizes2, model__size3=sizes3, batch_size=batch_sizes, epochs=epochss)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(t2d_X_train, t2d_y_train, validation_data=(t2d_X_test, t2d_y_test))

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def start():
    samplers = [RandomUnderSampler(sampling_strategy='majority'), RandomOverSampler(sampling_strategy='all')]
    df, t2d_X_train, t2d_X_test, t2d_y_train, t2d_y_test, ibd_X_train, ibd_X_test, ibd_y_train, ibd_y_test, cad_X_train, cad_X_test, cad_y_train, cad_y_test, ckd_X_train, ckd_X_test, ckd_y_train, ckd_y_test, n_features, label_count  = load_data(sampler=samplers[0])

    model = create_model(activation='relu', dropout_rate=0.0, init_mode='he_normal', kernel_regularizer_val1=1e-6, kernel_regularizer_val2=1e-5,
                 bias_regularizer_val=1e-5, activity_regularizer_val=1e-6, label_count=label_count, n_features=n_features,
                 opt_type=1, learning_rate=0.0000001, dropout=True, batchN=True, size1=4, size2=4, size3=4)

    X_val = []
    y_val = []
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    callbacks_list = []
    fit_model(model, t2d_X_train, t2d_y_train, X_val, y_val, epochs=32, batch_size=32, verbose=1, callbacks=callbacks_list)
    # save_model(model, "SpeciesCKD", "3x12294_500_279_0.5")
    # model = load_model(label_count, "SpeciesT2D", "3x12294_500_279_0.5")
    result = eval_model_adv(model, t2d_X_test, t2d_y_test)
    print(result)
    gc.collect()


def fit_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, verbose, callbacks):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose,
              callbacks=callbacks)  # validation_data=(X_val, y_val),


def create_model(activation, dropout_rate, init_mode, kernel_regularizer_val1, kernel_regularizer_val2,
                 bias_regularizer_val, activity_regularizer_val, label_count, n_features, opt_type,
                 learning_rate, dropout, batchN, size1, size2, size3):
    model = Sequential()
    if opt_type == 1:
        opt = keras.optimizers.Adam(
            learning_rate=learning_rate)
    else:
        opt = tf.keras.optimizers.experimental.SGD(
            learning_rate=learning_rate)

    sizes = [size1, size2, size3]
    inputlayer = True
    i = -1
    while i < 2:
        i = i+1
        create_layer(model=model, size=sizes[i], activation=activation, dropout=dropout,
                     dropout_rate=dropout_rate, batchN=batchN, inputlayer=inputlayer, init_mode=init_mode,
                     kernel_regularizer_val1=kernel_regularizer_val1, kernel_regularizer_val2=kernel_regularizer_val2,
                     bias_regularizer_val=bias_regularizer_val, activity_regularizer_val=activity_regularizer_val,
                     n_features=n_features)
        inputlayer = False
    model.add(Dense(label_count, activation="sigmoid"))
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_layer(model, size, activation, dropout, dropout_rate, batchN, inputlayer, init_mode,
                 kernel_regularizer_val1, kernel_regularizer_val2, bias_regularizer_val, activity_regularizer_val,
                 n_features):
    if inputlayer:
        model.add(Dense(size, activation=activation, kernel_initializer=init_mode,
                        kernel_regularizer=tf.keras.regularizers.L1L2(l1=kernel_regularizer_val1,
                                                                      l2=kernel_regularizer_val2),
                        bias_regularizer=tf.keras.regularizers.L2(bias_regularizer_val),
                        activity_regularizer=tf.keras.regularizers.L2(activity_regularizer_val),
                        input_shape=(n_features,)))
    else:
        model.add(Dense(size, activation=activation, kernel_initializer=init_mode,
                        kernel_regularizer=tf.keras.regularizers.L1L2(l1=kernel_regularizer_val1,
                                                                      l2=kernel_regularizer_val2),
                        bias_regularizer=tf.keras.regularizers.L2(bias_regularizer_val),
                        activity_regularizer=tf.keras.regularizers.L2(activity_regularizer_val)))
    if dropout:
        model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    if batchN:
        model.add(tf.keras.layers.BatchNormalization())


def load_data(sampler):
    df = pd.read_csv('./train_combined_Species.csv')
    df.drop(df.columns[0], axis=1, inplace=True)
    inputvalues = df.drop(['label'], axis=1)
    outputvalues = df['label']
    X, y = inputvalues.values, outputvalues.values
    X = X.astype('float32')
    y = y.reshape(-1, 1)
    y = OneHotEncoder(categories=[['0', '1', '2', '3', '4']], sparse=False).fit_transform(y)

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    # X = preprocessing.normalize(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    t2d_X_train, t2d_y_train, ibd_X_train, ibd_y_train, cad_X_train, cad_y_train, ckd_X_train, ckd_y_train = makedata(X_train, y_train)
    t2d_X_test, t2d_y_test, ibd_X_test, ibd_y_test, cad_X_test, cad_y_test, ckd_X_test, ckd_y_test = makedata(X_test, y_test)

    t2d_X_train, t2d_y_train = sampler.fit_resample(t2d_X_train, t2d_y_train)

    ibd_X_train, ibd_y_train = sampler.fit_resample(ibd_X_train, ibd_y_train)

    cad_X_train, cad_y_train = sampler.fit_resample(cad_X_train, cad_y_train)

    ckd_X_train, ckd_y_train = sampler.fit_resample(ckd_X_train, ckd_y_train)

    n_features_ = t2d_X_train.shape[1]

    '''
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0, random_state=1234)
    x_val = []
    y_val = []
    # print(Counter(y_train))
    x_train, y_train = sampler.fit_resample(x_train, y_train)
    # print(Counter(y_train))
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    n_features_ = x_train.shape[1]
    return df_, x_train, x_val, x_test, y_train, y_val, y_test, n_features_, 1
    '''

    return df, t2d_X_train, t2d_X_test, t2d_y_train, t2d_y_test, ibd_X_train, ibd_X_test, ibd_y_train, ibd_y_test, cad_X_train, cad_X_test, cad_y_train, cad_y_test, ckd_X_train, ckd_X_test, ckd_y_train, ckd_y_test, n_features_, 1


def makedata(X, y):
    neg_y = []
    neg_X = []

    t2d_y = []
    t2d_X = []

    ibd_y = []
    ibd_X = []

    cad_y = []
    cad_X = []

    ckd_y = []
    ckd_X = []

    for idx, row in enumerate(y):
        if y[idx][0] == 1:
            neg_y.append(0)
            neg_X.append(X[idx])
        elif y[idx][1] == 1:
            t2d_y.append(1)
            t2d_X.append(X[idx])
        elif y[idx][2] == 1:
            ibd_y.append(1)
            ibd_X.append(X[idx])
        elif y[idx][3] == 1:
            cad_y.append(1)
            cad_X.append(X[idx])
        elif y[idx][4] == 1:
            ckd_y.append(1)
            ckd_X.append(X[idx])

    neg_y = np.array(neg_y)
    neg_X = np.array(neg_X)

    t2d_y = np.array(t2d_y)
    t2d_X = np.array(t2d_X)
    t2d_y = np.concatenate([neg_y, t2d_y])
    t2d_X = np.concatenate([neg_X, t2d_X])

    ibd_y = np.array(ibd_y)
    ibd_X = np.array(ibd_X)
    ibd_y = np.concatenate([neg_y, ibd_y])
    ibd_X = np.concatenate([neg_X, ibd_X])

    cad_y = np.array(cad_y)
    cad_X = np.array(cad_X)
    cad_y = np.concatenate([neg_y, cad_y])
    cad_X = np.concatenate([neg_X, cad_X])

    ckd_y = np.array(ckd_y)
    ckd_X = np.array(ckd_X)
    ckd_y = np.concatenate([neg_y, ckd_y])
    ckd_X = np.concatenate([neg_X, ckd_X])

    return t2d_X, t2d_y, ibd_X, ibd_y, cad_X, cad_y, ckd_X, ckd_y


def save_model(model, checkpointfolder, checkpointname):
    model.save_weights('./' + checkpointfolder + '/' + checkpointname)


def load_model(label_count, checkpointfolder, checkpointname):
    model = create_model(label_count)
    model.load_weights('./' + checkpointfolder + '/' + checkpointname)
    return model


def eval_model_adv(model, X_test, y_test):
    stats = []

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    result = {
        "type": "all",
        "acc": acc
    }
    stats.append(result)

    neg_y_test = []
    neg_X_test = []

    pos_y_test = []
    pos_X_test = []

    for idx, row in enumerate(y_test):
        if (y_test[idx] == 0):
            neg_y_test.append(y_test[idx])
            neg_X_test.append(X_test[idx])
            # print(y_test[idx])
            # print(X_test[idx])
            # print(model(X_test[idx].reshape(1, 12711)))
        else:
            pos_y_test.append(y_test[idx])
            pos_X_test.append(X_test[idx])
            # print(y_test[idx])
            # print(X_test[idx])
            # print(model(X_test[idx].reshape(1, 12711)))

    neg_y_test = np.array(neg_y_test)
    neg_X_test = np.array(neg_X_test)
    loss, acc = model.evaluate(neg_X_test, neg_y_test, verbose=1)
    result = {
        "type": "neg",
        "acc": acc
    }
    stats.append(result)

    pos_y_test = np.array(pos_y_test)
    pos_X_test = np.array(pos_X_test)
    loss, acc = model.evaluate(pos_X_test, pos_y_test, verbose=1)
    result = {
        "type": "pos",
        "acc": acc
    }
    stats.append(result)
    return stats


i = 15
while i > 0:
    start()
    i = i - 1


#start()
#hyper_grid_search()
