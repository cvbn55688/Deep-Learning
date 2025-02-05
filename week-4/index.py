import math

def mse_function(outputs, expects):
    output_num = len(outputs)
    error_sum = 0
    for i in range(output_num):
        error_sum += (outputs[i] - expects[i]) ** 2
    return error_sum / output_num

def binary_cross_entropy_function(outputs, expects):
    output_num = len(outputs)
    error_sum = 0
    for i in range(output_num):
        error_sum += expects[i] * math.log(outputs[i]) + (1 - expects[i]) * math.log(1 - outputs[i])
    return -error_sum

def categorical_cross_entropy_function(outputs, expects):
    output_num = len(outputs)
    error_sum = 0
    for i in range(output_num):
        error_sum += expects[i] * math.log(outputs[i])
    return -error_sum

class Network:
    def __init__(self, all_weights, activations):
        self.all_weights = all_weights
        self.activations = activations

    def activation_functions(self, raw_outputs, activation):
        modified_outputs = []
        if activation == "linear":
            return raw_outputs 
        
        elif activation == "relu":

            for raw_output in raw_outputs:
                modified_outputs.append(raw_output if raw_output > 0 else 0)
            return modified_outputs
        
        elif activation == "sigmoid":
            for raw_output in raw_outputs:
                 modified_outputs.append(1 / (1 + math.exp(-raw_output)))
            return modified_outputs
        
        elif activation == "softmax":
            for raw_output in raw_outputs:
                modified_outputs.append(math.exp(raw_output))

            modified_outputs_sum = sum(modified_outputs)
            for i in range(len(modified_outputs)):
                modified_outputs[i] = modified_outputs[i] / modified_outputs_sum 
            return modified_outputs 

    def calculate_output(self, inputs, weights_for_layer, bias=1):
        output = []
        output_num = len(weights_for_layer[0])
        for _ in range(output_num):
            output.append(0)

        for j in range(len(weights_for_layer)):
            target_input = bias if j + 1 == len(weights_for_layer) else inputs[j]
            weights_for_node = weights_for_layer[j]
            for i in range(len(weights_for_node)):
                weight = weights_for_node[i]
                output[i] += target_input * weight

        return output

    def forward(self, inputs):
        for layer_index in range(len(self.all_weights)):
            inputs = self.activation_functions(self.calculate_output(inputs, self.all_weights[layer_index]), self.activations[layer_index])

        return inputs

print('----- Model 1 -----')
network1 = Network([[[0.5, 0.6], [0.2, -0.6], [0.3, 0.25]], [[0.8, 0.4], [-0.5, 0.5], [0.6, -0.25]]], ['relu', 'linear'])
print(f'[1-1] Total loss: {mse_function(network1.forward([1.5, 0.5]), [0.8, 1])}')
print(f'[1-2] Total loss: {mse_function(network1.forward([0, 1]), [0.5, 0.5])}')


print('----- Model 2 -----')
network2 = Network([[[0.5, 0.6], [0.2, -0.6], [0.3, 0.25]], [[0.8], [0.4], [-0.5]]], ['relu', 'sigmoid'])
print(f'[2-1] Total loss: {binary_cross_entropy_function(network2.forward([0.75, 1.25]), [1])}')
print(f'[2-2] Total loss: {binary_cross_entropy_function(network2.forward([-1, 0.5]), [0])}')

print('----- Model 3 -----')
network3 = Network([[[0.5, 0.6], [0.2, -0.6], [0.3, 0.25]], [[0.8, 0.5, 0.3], [-0.4, 0.4, 0.75], [0.6, 0.5, -0.5]]], ['relu', 'sigmoid'])
print(f'[3-1] Total loss: {binary_cross_entropy_function(network3.forward([1.5, 0.5]), [1, 0, 1])}')
print(f'[3-2] Total loss: {binary_cross_entropy_function(network3.forward([0, 1]), [1, 1, 0])}')

print('----- Model 4 -----')
network4 = Network([[[0.5, 0.6], [0.2, -0.6], [0.3, 0.25]], [[0.8, 0.5, 0.3], [-0.4, 0.4, 0.75], [0.6, 0.5, -0.5]]], ['relu', 'softmax'])
print(f'[4-1] Total loss: {categorical_cross_entropy_function(network4.forward([1.5, 0.5]), [1, 0, 0])}')
print(f'[4-2] Total loss: {categorical_cross_entropy_function(network4.forward([0, 1]), [0, 0, 1])}')
