import csv
import network
import torch
from torch.utils.data import Dataset, DataLoader

class GenderHeightWeightDataset(Dataset):
    def __init__(self, csv_file):       
        reader = csv.reader(open(csv_file))
        next(reader)

        inputs = []
        expects = []

        for row in reader:
            gender = 1 if row[0] == "Male" else 0
            height = float(row[1])
            weight = float(row[2])

            inputs.append((gender * 2) + height)
            expects.append([weight])

        mean_input = sum(inputs) / len(inputs)
        std_input = (sum([(x - mean_input) ** 2 for x in inputs]) / len(inputs)) ** 0.5
        inputs = [[(x - mean_input) / std_input] for x in inputs]

        self.data = torch.tensor(inputs)
        self.labels = torch.tensor(expects)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

print('\n============ Task 1 ============')
dataset = GenderHeightWeightDataset("gender-height-weight.csv")
dataloader = DataLoader(dataset, batch_size = 10, shuffle = True)
model = network.Network([1, 10, 1], ['relu', 'linear'])
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
criterion = torch.nn.MSELoss()

print("----- before training -----")
mae_sum = 0
input_len = 0
for inputs, targets in dataloader:
    outputs = model(inputs)
    mae_sum += torch.sum(torch.abs(outputs - targets)).item()
    input_len += len(inputs)

mae_loss = mae_sum / input_len
print(f"avg error: {mae_loss} lbs")  

#  training
for epoch in range(20):
    for inputs, targets in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("----- after training -----")
mae_sum = 0
input_len = 0
for inputs, targets in dataloader:
    outputs = model(inputs)
    mae_sum += torch.sum(torch.abs(outputs - targets)).item()
    input_len += len(inputs)

mae_loss = mae_sum / input_len
print(f"avg error: {mae_loss} lbs")  
