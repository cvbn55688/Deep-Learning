import csv
import network
import helper

print('\n============ Task 1 ============')

reader = csv.reader(open("gender-height-weight.csv"))
next(reader)

inputs = []
expects = []

for row in reader:
    gender = 1 if row[0] == "Male" else 0
    height = float(row[1])
    weight = float(row[2])

    inputs.append((gender * 2) + height)
    expects.append(weight)

mean_input = sum(inputs) / len(inputs)
std_input = (sum([(x - mean_input) ** 2 for x in inputs]) / len(inputs)) ** 0.5
inputs = [(x - mean_input) / std_input for x in inputs]

network1 = network.Network([
    {
        'node_weights': helper.random_weights(1, 10),
        'bias_weights': helper.random_biases_weights(10),
        'activation': 'sigmoid'
    },
    {
        'node_weights': helper.random_weights(10, 5),
        'bias_weights': helper.random_biases_weights(5),
        'activation': 'sigmoid'
    },
    {
        'node_weights': helper.random_weights(5, 1),
        'bias_weights': helper.random_biases_weights(1),
        'activation': 'linear'
    }
])

# lazy to write a function XD
mae_sum = 0
for input, expect in zip(inputs, expects):
    output = network1.forward([input])
    mae_sum += abs(output[0] - expect)  

mae_loss = mae_sum / len(inputs)
print("----- before traning -----")
print(f"avg error: {mae_loss} lbs")  


learning_rate = 0.001
epochs = 20

for i in range(epochs):
    for input, expect in zip(inputs, expects):
        output = network1.forward([input])
        losses = helper.mse_derivative(output, [expect])
        network1.backward(losses)
        network1.zero_grad(learning_rate)

mae_sum = 0
for input, expect in zip(inputs, expects):
    output = network1.forward([input])
    mae_sum += abs(output[0] - expect)  

mae_loss = mae_sum / len(inputs)
print("\n----- after traning -----")
print(f"avg error: {mae_loss} lbs")  
