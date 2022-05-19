import torch
from torch.utils.data import ConcatDataset, TensorDataset, DataLoader

datasets = []
for i in range(3):
    datasets.append(TensorDataset(torch.arange(i*10, (i+1)*10)))
    print(len(datasets[i]))

dataset = ConcatDataset(datasets)
loader = DataLoader(
    dataset,
    shuffle=False,
    num_workers=0,
    batch_size=3,
)

for data in loader:
    print(data)