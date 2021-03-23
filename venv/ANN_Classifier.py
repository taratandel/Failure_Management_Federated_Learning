from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import math


class MLPClassifierOverride(MLPClassifier):
    coef = []
    intercep = []
    layer_no = 0

    # Overriding _init_coef method
    def _init_coef(self, fan_in, fan_out, dtype):
        # Use the initialization method recommended by
        # Glorot et al.
        layer_no = self.layer_no
        factor = 6.
        if self.activation == 'logistic':
            factor = 2.
        init_bound = np.sqrt(factor / (fan_in + fan_out))

        # Generate weights and bias:
        coef_init = self._random_state.uniform(-init_bound, init_bound,
                                               (fan_in, fan_out))
        intercept_init = self._random_state.uniform(-init_bound, init_bound,
                                                    fan_out)
        if self.coef is not None:
            coef_init = self.coef[layer_no]
            intercept_init = self.intercep[layer_no]
        else:
            coef_init = coef_init.astype(dtype, copy=False)
            intercept_init = intercept_init.astype(dtype, copy=False)
        self.layer_no = self.layer_no + 1
        return coef_init, intercept_init


def trainANN(df, epochs, M, coef, intercept):
    windows = df
    #            ---------------------------------------- NAN PROCESSING ----------------------------------------
    # Identify the Transmitted power features
    tx = ['txMaxAN-2', 'txminAN-2', 'txMaxBN-2', 'txminBN-2', 'txMaxAN-1', 'txminAN-1',
          'txMaxBN-1', 'txminBN-1', 'txMaxAN', 'txminAN', 'txMaxBN', 'txminBN']
    # Fill the Nan value in the Tx power features with 100
    windows[tx] = windows[tx].fillna(100)
    # Fill the Nan value in the Rx power features with -150 We use two different values because in this problem the
    # Tx power and the Rx power have two different range of values, when we will normalize the data (Some steps
    # below) if the added values are too big this can eliminate the differences in the data values
    # ------------------------------------------------------------------------------------------------

    #           ---------------------------------------- LABELS PROCESSING ----------------------------------------
    # Put the labels inside a variable
    y = windows[windows.columns.values[-6:]].to_numpy()
    windows = windows.fillna(-150)

    #         ----------------------------------------- CLEAN THE FEATURES -----------------------------------------

    # Collect the columns name of the input
    columns = windows.columns.values
    # Now variable columns is a list containing all the names of the features
    # The variable is used to address and drop some features not useful in the training and cross-validation phase
    # Delete the columns that are not useful
    windows = windows.drop(columns=columns[:8])
    windows = windows.drop(columns=columns[-6:])
    # Delete the 'label' columns because we can't train a model using it as an input data
    # windows = windows.drop(columns=['label'])
    # After the elimination of the useless columns our data are described by 35 features
    # Transform the input data into an 2D-array
    X = windows.to_numpy()

    # ------------------------------------------------------------------------------------------------
    # provide to the object the data that will split
    # skf.get_n_splits(X, y)
    # Instantiate the scaler element
    scaler = StandardScaler()

    # Hyperparameters to set according to the model selection result
    # if the hyperparameters are given the model selection part must be commented as reported in the previous comments
    best_layers = 3
    best_neurons = 100
    best_activation = "tanh"

    # Scale the data
    X_train = scaler.fit_transform(X)  # Scale the train data as (value - mean) / std
    # X_test = scaler.transform(X_test)  # scale the test data as (value - mean_train) / std_train

    # Create the object of the best model according to Cross-Validation
    size = (best_neurons,) * best_layers  # Create the structure of the Artificial Neural Network
    if M == math.inf:
        M = 'auto'
    ann = MLPClassifierOverride(hidden_layer_sizes=size, activation=best_activation,
                                solver='adam', learning_rate='invscaling', max_iter=epochs, batch_size=M)
    ann.coef = coef
    ann.intercep = intercept
    ann.fit(X_train, y)

    return ann