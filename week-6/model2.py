import csv
import network
import helper

print('\n============ Task 2 ============')

inputs = []
expects = []
with open("titanic.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
            # 如果有空值就不要
            if row['Sex'] and row['Pclass'] and row['Age'] and row['Fare'] and row['Survived']:
                gender = 1 if row['Sex'].strip().lower() == "female" else 0

                pclass = float(row['Pclass'])
                age = float(row['Age'])
                fare = float(row['Fare'])
                survived = float(row['Survived'])

                inputs.append([gender, pclass, age, fare])
                expects.append(survived)

# 標準化
mean_age = sum(row[2] for row in inputs) / len(inputs)
std_age = (sum((row[2] - mean_age) ** 2 for row in inputs) / len(inputs)) ** 0.5

mean_fare = sum(row[3] for row in inputs) / len(inputs)
std_fare = (sum((row[3] - mean_fare) ** 2 for row in inputs) / len(inputs)) ** 0.5

for row in inputs:
    row[2] = (row[2] - mean_age) / std_age
    row[3] = (row[3] - mean_fare) / std_fare


network2 = network.Network([
    {
        'node_weights': helper.random_weights(4, 10),
        'bias_weights': helper.random_biases_weights(10),
        'activation': 'relu'
    },
    {
        'node_weights': helper.random_weights(10, 3),
        'bias_weights': helper.random_biases_weights(3),
        'activation': 'relu'
    },
    {
        'node_weights': helper.random_weights(3, 1),
        'bias_weights': helper.random_biases_weights(1),
        'activation': 'sigmoid'
    }
])


correct_count = 0
threshold = 0.5 
for input, expect in zip(inputs, expects):
    output = network2.forward(input)[0]
    survival_status = 0 
    if output > threshold: 
        survival_status = 1 
    if survival_status == expect: 
        correct_count += 1

correct_rate = correct_count/len(expects) * 100
print("----- before traning -----")
print(f"correct rate: {correct_rate}%")  



learning_rate = 0.001
epochs = 100

for i in range(epochs):
    for input, expect in zip(inputs, expects):
        output = network2.forward(input)
        losses = helper.binary_cross_entropy_derivative(output, [expect])
        network2.backward(losses)
        network2.zero_grad(learning_rate)

correct_count = 0
threshold = 0.5 
for input, expect in zip(inputs, expects):
    output = network2.forward(input)[0]
    survival_status = 0 
    if output > threshold: 
        survival_status = 1 
    if survival_status == expect: 
        correct_count += 1

correct_rate = correct_count/len(expects) * 100
print("\n----- after traning -----")
print(f"correct rate: {correct_rate}%")  
