import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from tensorflow import keras
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import classification_report


# search for optimal hyperparameter
def hyper_grid_search():
    # creating the Keras Model
    model = KerasClassifier(model=create_model, verbose=1)

    # Load the dataset
    samplers = [RandomUnderSampler(sampling_strategy='majority'), RandomOverSampler(sampling_strategy='all')]
    df, X_train, X_test, y_train, y_test, n_features = load_data(sampler=samplers[1])

    # Hyperparameter for the Model
    label_count = [5]
    n_features = [n_features]
    batch_sizes = [64]
    epochss = [60, 90] # 74
    activations = ['relu'] # activations = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    init_mode = ['glorot_normal'] # init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    dropout_rates = [0.1, 0.2]
    kernel_regularizer_val1s = [1e-6, 1e-5] #[1e-6, 1e-5, 1e-4, 1e-3]
    kernel_regularizer_val2s = [1e-5, 1e-4] #[1e-5, 1e-4, 1e-3, 1e-2]
    bias_regularizer_vals = [1e-5, 1e-4] #[1e-5, 1e-4, 1e-3, 1e-2]
    activity_regularizer_vals = [1e-6, 1e-5] #[1e-6, 1e-5, 1e-4, 1e-3]
    opt_types = [1]
    learning_rates = [1e-05, 1e-06, 1e-07]  # [0.0000001, 0.000001, 0.00001, 0.0001]
    dropouts = [True]
    batchNs = [True]
    sizes1 = [64]  # [4, 8, 16, 32]
    sizes2 = [32]
    sizes3 = [4]
    param_grid = dict(model__activation=activations, model__dropout_rate=dropout_rates, model__init_mode=init_mode,
                      model__kernel_regularizer_val1=kernel_regularizer_val1s,
                      model__kernel_regularizer_val2=kernel_regularizer_val2s,
                      model__bias_regularizer_val=bias_regularizer_vals,
                      model__activity_regularizer_val=activity_regularizer_vals,
                      model__label_count=label_count, model__n_features=n_features,
                      model__opt_type=opt_types, model__learning_rate=learning_rates, model__dropout=dropouts,
                      model__batchN=batchNs,
                      model__size1=sizes1, model__size2=sizes2, model__size3=sizes3, batch_size=batch_sizes,
                      epochs=epochss)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose=3)
    grid_result = grid.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=0)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def trainmodel(iters):
    samplers = [RandomUnderSampler(sampling_strategy='majority'), RandomOverSampler(sampling_strategy='all')]
    df, X_train, X_test, y_train, y_test, n_features = load_data(sampler=samplers[1])
    i = iters
    while i > 0:
        i = i - 1
        model = create_model(activation='relu', dropout_rate=0.1, init_mode='he_normal', kernel_regularizer_val1=1e-6,
                             kernel_regularizer_val2=1e-5,
                             bias_regularizer_val=1e-5, activity_regularizer_val=1e-6, label_count=5,
                             n_features=n_features,
                             opt_type=1, learning_rate=1e-06, dropout=True, batchN=True, size1=64, size2=32, size3=5)

        # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        callbacks_list = []
        model.fit(X_train, y_train, epochs=75, batch_size=64, verbose=0, callbacks=callbacks_list)
        eval_model(model, X_test, y_test)


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

    sizes = [size1, size1, size2, size2, size3, size3]
    inputlayer = True
    i = -1
    while i < 2:
        i = i + 1
        create_layer(model=model, size=sizes[i], activation=activation, dropout=dropout,
                     dropout_rate=dropout_rate, batchN=batchN, inputlayer=inputlayer, init_mode=init_mode,
                     kernel_regularizer_val1=kernel_regularizer_val1, kernel_regularizer_val2=kernel_regularizer_val2,
                     bias_regularizer_val=bias_regularizer_val, activity_regularizer_val=activity_regularizer_val,
                     n_features=n_features)
        inputlayer = False
    model.add(Dense(label_count, activation="sigmoid"))
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
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


def eval_model(model, X_test, y_test):
    Y_test = np.argmax(y_test, axis=1)  # Convert one-hot to index
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    print(classification_report(Y_test, y_pred))


def load_data(sampler):
    df = pd.read_csv('../../ModelKeras/train_combined_Species.csv')
    df.drop(df.columns[0], axis=1, inplace=True)
    df = df.fillna(0).drop_duplicates()

    inputvalues = df.drop(['label'], axis=1)
    outputvalues = df['label']

    scaler = MinMaxScaler()
    inputvalues = scaler.fit_transform(inputvalues)

    # pca = PCA(n_components=600)
    # inputvalues = pca.fit_transform(inputvalues)

    X, y = inputvalues, outputvalues.values
    X = X.astype('float32')
    y = y.reshape(-1, 1)
    y = OneHotEncoder(categories=[['0', '1', '2', '3', '4']], sparse=False).fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    X_train, y_train = sampler.fit_resample(X_train, y_train)
    n_features = X_train.shape[1]
    return df, X_train, X_test, y_train, y_test, n_features


trainmodel(5)
# hyper_grid_search()

# model.save_weights('./' + "3x12294_500_279_0.5")

# model = create_model(label_count)
# model.load_weights('./' + "3x12294_500_279_0.5")
