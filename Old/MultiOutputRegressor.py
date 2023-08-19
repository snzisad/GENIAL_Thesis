import numpy as np

class MultiOutputRegressor:
    def __init__(self, num_outputs, learning_rate=0.01, num_epochs=1000):
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
    
    def fit(self, X, y):
        # Add bias term to input data
        X = np.c_[np.ones(X.shape[0]), X]
        
        # Initialize weights randomly
        self.weights = np.random.rand(X.shape[1], self.num_outputs)
        
        # Gradient descent loop
        for i in range(self.num_epochs):
            # Calculate predictions and error
            y_pred = X.dot(self.weights)
            error = y_pred - y
            
            # Calculate gradients and update weights
            gradients = X.T.dot(error) / X.shape[0]
            self.weights -= self.learning_rate * gradients
    
    def predict(self, X):
        # Add bias term to input data
        X = np.c_[np.ones(X.shape[0]), X]
        
        # Make predictions
        y_pred = X.dot(self.weights)
        
        return y_pred
