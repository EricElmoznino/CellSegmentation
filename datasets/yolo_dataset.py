from torch.utils.data import Dataset
from torchvision import transforms as tr


class YoloDataset(Dataset):

    def __init__(self, dir, augment=False):
        self.data = []
