__copyright__ = "Copyright 2020, UPB-CAMPUS Research Center - Multimedia Lab"
__email__ = "liviu_daniel.stefan@upb.ro, cmihaigabriel@gmail.com"

from base import BaseDataLoader
from torchvision import transforms, datasets


class FakeLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, input_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.num_workers = num_workers
        self.training = training
        self.input_size = input_size

        self.data_transforms = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.dataset = datasets.ImageFolder(root=self.data_dir, transform=self.data_transforms)

        super().__init__(self.dataset, self.batch_size, self.shuffle, self.validation_split, self.num_workers)
