from sklearn.neural_network import MLPClassifier
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


def trainANN(X_train, y, epochs, M, coef, intercept):
    best_layers = 3
    best_neurons = 100
    best_activation = "tanh"

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