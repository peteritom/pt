import numpy as np
from random import random

# implement backpropagation
# implement gradiant descent
# implement train
# train with domain datasets
# make predictions

class MLP(object):

    """A Multilayer Perceptron class.
    """

    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2):
        """Constructor for the MLP. Takes the number of inputs,
            a variable number of hidden layers, and number of outputs
        Args:
            num_inputs (int): Number of inputs
            hidden_layers (list): A list of ints for the hidden layers
            num_outputs (int): Number of outputs
        """

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]

        # create random connection weights for the layers
        weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights

        # save activations and derivatives
        activations = []
        derivatives = []
        for i in range(len(layers)):
            activations.append(np.zeros(layers[i]))
        self.activations = activations
        for i in range(len(layers)-1):
            derivatives.append(np.zeros([layers[i], layers[i+1]]))
        self.derivatives = derivatives


    def forward_propagate(self, inputs):
        """Computes forward propagation of the network based on input signals.
        Args:
            inputs (ndarray): Input signals
        Returns:
            activations (ndarray): Output values
        """

        # the input layer activation is just the input itself
        activations = inputs
        self.activations[0] = inputs

        # iterate through the network layers
        for i, w in enumerate(self.weights):

            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            # apply sigmoid activation function
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations

        # return output layer activation
        return activations

    def back_propagate(self, error, verbose=False):
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T

            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print("Derivatives for W{}: {}".format(i, self.derivatives[i]))

        return error

    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate

    def train(self, inputs, targets, epochs, learning_rate):

        for i in range(epochs):

            sum_error = 0

            for input, target in zip(inputs, targets):
                # perform forward propagation
                output = self.forward_propagate(input)
                #print("Network activation: {}".format(output))

                # error calculate
                error = target - output

                # perform back propagation
                self.back_propagate(error)

                # apply gradient descent
                self.gradient_descent(learning_rate)

                sum_error += self._mse(target, output)
            
            # report error
            #print("Error: {} at epoch {}".format(sum_error/len(inputs), i))

    def _mse(self, target, output):
        return np.average((target - output)**2)



    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def _sigmoid(self, x):
        """Sigmoid activation function
        Args:
            x (float): Value to be processed
        Returns:
            y (float): Output
        """
        
        y = 1.0 / (1 + np.exp(-x))
        return y


if __name__ == "__main__":

    # create a Multilayer Perceptron
    mlp = MLP(2, [5], 1)

    # set random values for network's input
    #inputs = np.random.rand(mlp.num_inputs)
    
    inputs = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in inputs])

    #inputs = np.array([[0.1, 0.4]])
    #targets = np.array([[0.5]])

    # train mlp
    mlp.train(inputs, targets, 50, 0.4)

    # create dummy data

    input = np.array([0.3, 0.1])
    target = np.array([0.4])

    output = mlp.forward_propagate(input)

    print ()
    print ()
    print ("Our network believes that {} + {} is equal to {}".format(input[0], input[1], output[0]))

    
