import csv
import network
import torch
from torch.utils.data import Dataset, DataLoader


class TitanicDataset(Dataset):
    def __init__(self, csv_file):
        inputs = []
        expects = []

        with open(csv_file, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['Sex'] and row['Pclass'] and row['Age'] and row['Fare'] and row['Survived']:
                    gender = 1 if row['Sex'].strip().lower() == "female" else 0
                    pclass = float(row['Pclass'])
                    age = float(row['Age'])
                    fare = float(row['Fare'])
                    survived = float(row['Survived'])

                    inputs.append([gender, pclass, age, fare])
                    expects.append([survived])

        mean_age = sum(row[2] for row in inputs) / len(inputs)
        std_age = (sum((row[2] - mean_age) ** 2 for row in inputs) / len(inputs)) ** 0.5

        mean_fare = sum(row[3] for row in inputs) / len(inputs)
        std_fare = (sum((row[3] - mean_fare) ** 2 for row in inputs) / len(inputs)) ** 0.5

        for row in inputs:
            row[2] = (row[2] - mean_age) / std_age
            row[3] = (row[3] - mean_fare) / std_fare

        self.data = torch.tensor(inputs)
        self.labels = torch.tensor(expects)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

print('\n============ Task 2 ============')
dataset = TitanicDataset("titanic.csv")
dataloader = DataLoader(dataset, batch_size = 10)
model = network.Network([4, 10, 3, 1], ['relu', 'relu', 'sigmoid'])
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05)
criterion = torch.nn.BCELoss()

print("----- before training -----")
correct_count = 0
input_len = 0
threshold = 0.5 
for inputs, expects in dataloader:
    outputs = model(inputs)
    predicted = (outputs > threshold).float()
    correct_count += torch.sum(predicted == expects).item()
    input_len += expects.size(0)

correct_rate = correct_count/input_len * 100
print(f"correct rate: {correct_rate}%")  

#  training
for epoch in range(100):
    for inputs, expects in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, expects)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


print("----- after training -----")
correct_count = 0
input_len = 0
threshold = 0.5 
for inputs, expects in dataloader:
    outputs = model(inputs)
    predicted = (outputs > threshold).float()
    correct_count += torch.sum(predicted == expects).item()
    input_len += expects.size(0)

correct_rate = correct_count / input_len * 100
print(f"correct rate: {correct_rate}%")  