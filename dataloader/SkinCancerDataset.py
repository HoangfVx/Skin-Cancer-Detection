from torch.utils.data import Dataset, DataLoader

class SkinCancerDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        super().__init__()
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __getitem__(self, idx):
        img, _ = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
        
    def __len__(self):
        return len(self.images)