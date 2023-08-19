import numpy as np


def DNN(param, output_size, X):
    input_size = 9
    num_hidden_layers = 5
    num_neuron_size = 10
    weights = []
    param = np.array(param)
    last_used_pos = 0
    
    for i in range(num_hidden_layers+2):
        # input layer
        if i == 0:
            total_neuron = num_neuron_size*input_size
            weights.append(np.reshape(param[last_used_pos:last_used_pos+total_neuron], (-1, input_size)))
            # weights.append(np.random.rand(num_neuron_size, input_size))
        
        # output layer
        elif i == num_hidden_layers+1:
            total_neuron = output_size*num_neuron_size
            weights.append(np.reshape(param[last_used_pos:last_used_pos+total_neuron], (-1, num_neuron_size)))
            # weights.append(np.random.rand(output_size, num_neuron_size))
            
        # Hidden layers  
        else:
            total_neuron = num_neuron_size*num_neuron_size
            weights.append(np.reshape(param[last_used_pos:last_used_pos+total_neuron], (-1, num_neuron_size)))
            # weights.append(np.random.rand(num_neuron_size, num_neuron_size))
            
        last_used_pos += total_neuron
        
    biases = param[last_used_pos:last_used_pos+num_hidden_layers+1]
    print(biases)
    
    prev_layer_input = sigmoid(X.dot(weights[0].T)+biases[0])
    for i in range(1, num_hidden_layers):
        prev_layer_input = sigmoid(prev_layer_input.dot(weights[i])+biases[i])
    
    activation_weights = sigmoid(prev_layer_input.dot(weights[-1].T)+biases[-1])
    
    return activation_weights

    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

num_param = 10*(9+5*10+27)+5+1
param = np.random.rand(num_param)
X = np.random.rand(9)
print(DNN(param, 27, X))

# 10*(9+5*10+3)+5+1


# A = np.array([1,2,3,4,5,6])
# B = np.reshape(A, (-1, 3))
# print(B)




