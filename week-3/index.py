class Network:
    def __init__(self, all_weights):
        self.all_weights = all_weights

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
        for weights_for_layer in self.all_weights:
            inputs = self.calculate_output(inputs, weights_for_layer)

        return inputs

print('----- Model 1 -----')
network1 = Network([[[0.5, 0.6], [0.2, -0.6], [0.3, 0.25]], [[0.8], [0.4], [-0.5]]])
print(network1.forward([1.5, 0.5]))
print(network1.forward([0, 1]))

print('----- Model 2 -----')
network2 = Network([[[0.5, 0.6], [1.5, -0.8], [0.3, 1.25]], [[0.6], [-0.8], [0.3]], [[0.5, -0.4], [0.2, 0.5]]])
print(network2.forward([0.75, 1.25]))
print(network2.forward([-1, 0.5]))
