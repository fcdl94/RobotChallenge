import torch


class DoubleDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataset1, dataset2, transforms=None):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.transform = transforms
    
    def __getitem__(self, index):
        
        return self.dataset1[index], self.dataset2[index]
    
    def dataset_len(self, dataset=1):
        if dataset == 1:
            return len(self.dataset1)
        else:
            return len(self.dataset2)
    
    def __len__(self):
        return min(len(self.dataset1), len(self.dataset2))

