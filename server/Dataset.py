import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np

class WineDataset(Dataset):

    def __init__(self):
        # Data loading
        data_path = r'E:\Masters\Semester 4\Housing price prediction\server\wine.csv'
        data = np.loadtxt(data_path, delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(data[:, 1:])
        self.y = torch.from_numpy(data[:, [0]])
        self.samples = data.shape[0]

    # Dataset indexing
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # len(dataset)
    def __len__(self):
        return self.samples

'''if __name__ == '__main__':

    # Instantiate the dataset and dataloader
    dataset = WineDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

    num_epochs = 2
    total_samples = len(dataset)
    n_iterations = total_samples/4
    print(total_samples, n_iterations)
    
    for epoch in range(num_epochs):
        for i, (inputs, outputs) in enumerate(dataloader):
            if (i+1)%5 ==0:
                print(f'inputs: {inputs.shape }')'''

train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=torchvision.transforms.ToTensor(),  
                                           download=True)

train_loader = DataLoader(dataset=train_dataset, 
                                           batch_size=3, 
                                           shuffle=True)

dataiter = iter(train_loader)
data = next(dataiter)
inputs, targets = data
print(inputs.shape, targets.shape)