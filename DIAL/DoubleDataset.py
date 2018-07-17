import torch


class DoubleDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
    
    def __getitem__(self, index):
        if index >= len(self.dataset1):
            data_1 = self.dataset1[index - len(self.dataset1)]
        else:
            data_1 = self.dataset1[index]
        if index >= len(self.dataset2):
            data_2 = self.dataset2[index - len(self.dataset2)]
        else:
            data_2 = self.dataset2[index]
        
        return data_1, data_2

    def __len__(self):
        return max(len(self.dataset1), len(self.dataset2))

