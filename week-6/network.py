import math

class Network:
    def __init__(self, layers):
        self.layers = layers
        self.layer_outputs = []
        self.gradients = [] 

    def sigmoid(self, x):
        if x >= 0:
            z = math.exp(-x)
            return 1 / (1 + z)
        else:
            z = math.exp(x)
            return z / (1 + z)
            
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
        
    def activation_derivative_functions(self, original_outputs, activation):
        modified_outputs = []

        if activation == "linear":
            for i in range(len(original_outputs)):
                modified_outputs.append(1)
        
        elif activation == "relu":
            for i in range(len(original_outputs)):
                modified_outputs.append((1 if original_outputs[i] > 0 else 0) )
        
        elif activation == "sigmoid":
            for i in range(len(original_outputs)):
                 modified_outputs.append((original_outputs[i] * (1 - original_outputs[i])))

        return modified_outputs
        
    def calculate_output(self, inputs, weights_for_layer, biases):
        outputs = [0] * len(weights_for_layer[0])
        for j, node_weights in enumerate(weights_for_layer):
            for i, weight in enumerate(node_weights):
                outputs[i] += inputs[j] * weight
        for i in range(len(outputs)):
            outputs[i] += biases[i] 
        return outputs

    def forward(self, inputs):
        self.layer_outputs = []
        self.layer_outputs.append(inputs)
        for layer in self.layers:
            weights = layer['node_weights']
            biases = layer.get('bias_weights', [0] * len(weights[0]))
            activation = layer['activation']
            inputs = self.activation_functions(self.calculate_output(inputs, weights, biases), activation)
            self.layer_outputs.append(inputs)
        return inputs
    

    def backward(self, losses):
        current_loss = losses
        self.gradients = []

        # 從最後一層開始反向傳播
        for reversed_index in reversed(range(len(self.layers))):
            layer = self.layers[reversed_index]
            activation = layer['activation']
            original_output = self.layer_outputs[reversed_index + 1]
            prev_output = self.layer_outputs[reversed_index]


            # 計算 activation
            activation_derivative = self.activation_derivative_functions(original_output, activation)
            layer_loss = [a * b for a, b in zip(current_loss, activation_derivative)]

            # 權重梯度
            weight_gradients = []
            for j in range(len(layer['node_weights'])):
                weight_gradients.append([prev_output[j] * l for l in layer_loss])

            # bias 梯度
            bias_gradients =  1 * layer_loss
            self.gradients.insert(0, {'weight_gradients': weight_gradients, 'bias_gradients': bias_gradients})

            # 傳遞損失到前一層
            # bias 不用算回前一層
            new_loss = []
            for j in range(len(layer['node_weights'])):
                sum_loss = 0
                for k in range(len(layer['node_weights'][j])):
                    sum_loss += layer_loss[k] * layer['node_weights'][j][k]
                new_loss.append(sum_loss)
            current_loss = new_loss

    def zero_grad(self, learning_rate):
        for i, layer in enumerate(self.layers):
            gradients = self.gradients[i]
            
            for j in range(len(layer['node_weights'])):
                for k in range(len(layer['node_weights'][j])):
                    layer['node_weights'][j][k] -= learning_rate * gradients['weight_gradients'][j][k]
            
            for j in range(len(layer['bias_weights'])):
                layer['bias_weights'][j] -= learning_rate * gradients['bias_gradients'][j]
        return self.gradients