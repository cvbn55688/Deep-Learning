import math
import copy

def mse_function(outputs, expects):
    output_num = len(outputs)
    error_sum = 0
    for i in range(output_num):
        error_sum += (outputs[i] - expects[i]) ** 2
    return error_sum / output_num

def mse_derivative(outputs, expects):
    output_num = len(outputs)
    gradients = []
    for i in range(output_num):
        gradients.append((2 / output_num) * (outputs[i] - expects[i]))
    return gradients

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
        self.all_weights_gradients = copy.deepcopy(all_weights)
        self.activations = activations
        self.layer_value_records = []
        
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
        
    def activation_derivative_functions(self, original_outputs, error_changes, activation):
        modified_outputs = []

        if activation == "linear":
            for i in range(len(original_outputs)):
                modified_outputs.append(1 * error_changes[i])
        
        elif activation == "relu":
            for i in range(len(original_outputs)):
                modified_outputs.append((1 if original_outputs[i] > 0 else 0) * error_changes[i])
        
        elif activation == "sigmoid":
            for i in range(len(original_outputs)):
                 modified_outputs.append((original_outputs[i] * (1 - original_outputs[i])) *  error_changes[i])

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
        self.layer_value_records.append(inputs)
        for layer_index in range(len(self.all_weights)):
            inputs = self.activation_functions(self.calculate_output(inputs, self.all_weights[layer_index]), self.activations[layer_index])
            self.layer_value_records.append(inputs)
        return inputs
    
    def backward(self, error_changes):
        for i, activation in enumerate(reversed(self.activations)):
            index = len(self.activations) - i
            original_input = self.layer_value_records[index]
            last_inputs = self.layer_value_records[index - 1].copy()
            last_inputs.append(1) # for adding bais

            error_graidents = self.activation_derivative_functions(original_input, error_changes, activation)
            error_changes = []
            # This extra index is for layer_value_records which contain the input (x1, x2, ...)
            # So we need to index - 1
            for input_index, original_gradients in enumerate(self.all_weights_gradients[index - 1]):
                new_error_change = 0
                for j in range(len(original_gradients)):
                    # original_gradients[j] can directly change the self.all_weights_gradients
                    origin_weights = original_gradients[j]
                    new_error_change += origin_weights * error_graidents[j]
                    original_gradients[j] = last_inputs[input_index] * error_graidents[j]
                # To pervent bais append error_changes
                if(input_index < len(last_inputs) -1 ): error_changes.append(new_error_change)

    def zero_grad(self, learning_rate):
        for i in range(len(self.all_weights)):
            for j in range(len(self.all_weights[i])):
                for k in range(len(self.all_weights[i][j])):
                    self.all_weights[i][j][k] -= learning_rate * self.all_weights_gradients[i][j][k]
        return self.all_weights


print('======= Model 1 =======')
learning_rate = 0.01
expects = [0.8, 1]
network1 = Network([[[0.5, 0.6], [0.2, -0.6], [0.3, 0.25]], [[0.8], [-0.5], [0.6]], [[0.6, -0.3], [0.4, 0.75]]], ['relu', 'linear', 'linear'])

for i in range(1000):
    output = (network1.forward([1.5, 0.5]))
    error_changes = mse_derivative(output, expects)
    network1.backward(error_changes)
    updated_weights = network1.zero_grad(learning_rate)
    if(i == 0): 

        transposed_with_names = [
            {f"layer{i}: {list(zip(*layer))}"} for i, layer in enumerate(updated_weights)
        ]
        print('------ task 1 ------')
        print("NOTICE: My single layer contain bais's weight.\n")
        print(transposed_with_names)

print('------ task 2 ------')
print(f'Total Loss: {mse_function(network1.forward([1.5, 0.5]), expects)}')


print('\n\n======= Model 2 =======')
learning_rate = 0.1
expects = [1]
network2 = Network([[[0.5, 0.6], [0.2, -0.6], [0.3, 0.25]], [[0.8], [0.4], [-0.5]]], ['relu', 'sigmoid'])

for i in range(1000):
    output = (network2.forward([0.75, 1.25]))
    error_changes = mse_derivative(output, expects)
    network2.backward(error_changes)
    updated_weights = network2.zero_grad(learning_rate)
    if(i == 0): 
        print('------ task 1 ------')
        print("NOTICE: My single layer will contain bais's weight.\n")
        transposed_with_names = [
            {f"layer{i}: {list(zip(*layer))}"} for i, layer in enumerate(updated_weights)
        ]
        print(transposed_with_names)

print('------ task 2 ------')
print(f'Total Loss: {mse_function(network2.forward([0.75, 1.25]), expects)}')
