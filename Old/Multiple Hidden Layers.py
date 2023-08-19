import numpy as np

class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden_layers, num_hidden_neurons, num_outputs, learning_rate=0.01, num_epochs=1000):
        self.num_inputs = num_inputs
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_neurons = num_hidden_neurons
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
    
    def fit(self, X, y):
        # Initialize weights and biases randomly
        self.weights_hidden = []
        self.biases_hidden = []
        self.weights_output = np.random.rand(self.num_hidden_neurons[-1], self.num_outputs)
        self.biases_output = np.random.rand(1, self.num_outputs)
        
        for i in range(self.num_hidden_layers):
            if i == 0:
                self.weights_hidden.append(np.random.rand(self.num_inputs, self.num_hidden_neurons[i]))
            else:
                self.weights_hidden.append(np.random.rand(self.num_hidden_neurons[i-1], self.num_hidden_neurons[i]))
            self.biases_hidden.append(np.random.rand(1, self.num_hidden_neurons[i]))
        
        # Gradient descent loop
        for i in range(self.num_epochs):
            # Forward propagation
            hidden_layers_output = []
            for j in range(self.num_hidden_layers):
                if j == 0:
                    hidden_layer_output = self.sigmoid(X.dot(self.weights_hidden[j]) + self.biases_hidden[j])
                else:
                    hidden_layer_output = self.sigmoid(hidden_layers_output[-1].dot(self.weights_hidden[j]) + self.biases_hidden[j])
                hidden_layers_output.append(hidden_layer_output)
                
            output = self.sigmoid(hidden_layers_output[-1].dot(self.weights_output) + self.biases_output)
            
            # Backpropagation
            error_output = output - y
            delta_output = error_output * self.sigmoid_derivative(output)
            error_hidden = [delta_output.dot(self.weights_output.T)]
            delta_hidden = [error_hidden[-1] * self.sigmoid_derivative(hidden_layers_output[-1])]
            
            for j in range(self.num_hidden_layers-2, -1, -1):
                error_hidden.append(delta_hidden[-1].dot(self.weights_hidden[j+1].T))
                delta_hidden.append(error_hidden[-1] * self.sigmoid_derivative(hidden_layers_output[j]))
                
            delta_hidden.reverse()
            
            # Update weights and biases
            self.weights_output -= self.learning_rate * hidden_layers_output[-1].T.dot(delta_output)
            self.biases_output -= self.learning_rate * np.sum(delta_output, axis=0, keepdims=True)
            
            for j in range(self.num_hidden_layers):
                if j == 0:
                    self.weights_hidden[j] -= self.learning_rate * X.T.dot(delta_hidden[j])
                else:
                    self.weights_hidden[j] -= self.learning_rate * hidden_layers_output[j-1].T.dot(delta_hidden[j])
                self.biases_hidden[j] -= self.learning_rate * np.sum(delta_hidden[j], axis=0, keepdims=True)
    
    def predict(self, X):
        # Forward propagation
        hidden_layers_output = []
        for j in range(self.num_hidden_layers):
            if j == 0:
                hidden_layer_output = self.sigmoid(X.dot(self.weights_hidden[j]) + self.biases_hidden[j])
            else:
                hidden_layer_output = self.sigmoid(hidden_layers_output[-1].dot(self.weights_hidden[j]) + self.biases_hidden[j])
            hidden_layers_output.append(hidden_layer_output)

        output = self.sigmoid(hidden_layers_output[-1].dot(self.weights_output
