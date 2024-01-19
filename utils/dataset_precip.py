from torch.utils.data import Dataset
import torch.multiprocessing
import h5py
import numpy as np
import matplotlib.pyplot as plt
    
class precipitation_maps_masked_h5(Dataset):
    def __init__(self, in_file, num_input_images, num_output_images, mode="train", transform=None, use_timestamps=False):
        super(precipitation_maps_masked_h5, self).__init__()
        # The default sharing strategy is not supporten on mac
        torch.multiprocessing.set_sharing_strategy('file_system')

        self.file_name = in_file
        self.samples, _, _, _ = h5py.File(self.file_name, "r")[mode]["images"].shape
        self.samples = int(self.samples * 1)
        self.num_input = num_input_images
        self.num_output = num_output_images

        self.mode = mode
        self.use_timestamps = use_timestamps
        self.transform = transform
        self.dataset = None
        self.timestamps = None

    def __getitem__(self, index):
        # load the file here (load as singleton)
        if self.dataset is None:
            self.dataset = h5py.File(self.file_name, "r", rdcc_nbytes=1024**3)[self.mode]["images"]
        if self.timestamps is None and self.use_timestamps is True:
            self.timestamps = h5py.File(self.file_name, "r", rdcc_nbytes=1024**3)[self.mode]["timestamps"]
        imgs = np.array(self.dataset[index], dtype="float32")

        # add transforms
        if self.transform is not None:
            imgs = self.transform(imgs)
        _, h, w = imgs.shape
        # crop
        start_row = (h - 64) // 2
        start_col = (w - 64) // 2
        imgs = imgs[:,start_row:start_row+64, start_col:start_col+64]

        input_imgs = imgs[: self.num_input]
        target_imgs = imgs[self.num_input:len(imgs)]

        factor = 52.52
        input_imgs_sum = (np.sum(input_imgs, axis=0) * factor)[None,:]
        target_imgs_sum = (np.sum(target_imgs, axis=0) * factor)[None,:]

        thresholds = range(1,26,1)
        for i, t in enumerate(thresholds):
            input_mask = (input_imgs_sum >= t).astype("float32")
            input_masks = np.concatenate((input_masks, input_mask), axis=0) if i>0 else input_mask

            target_mask = (target_imgs_sum >= t).astype("float32")
            target_masks = np.concatenate((target_masks, target_mask), axis=0) if i>0 else target_mask

        if not self.use_timestamps:
            return input_imgs, input_masks, target_imgs, target_masks
        else:
            month = str(self.timestamps[index][self.num_input][0])[5:8]
            season = self.get_season(month)
            return input_imgs, input_masks, target_imgs, target_masks, season

    def __len__(self):
        return self.samples
    
    def get_season(self, str):
        months = {
            'JAN': (0, 0),
            'FEB': (1, 0),
            'MAR': (2, 1),
            'APR': (3, 1),
            'MAY': (4, 1),
            'JUN': (5, 2),
            'JUL': (6, 2),
            'AUG': (7, 2),
            'SEP': (8, 3),
            'OCT': (9, 3),
            'NOV': (10, 3),
            'DEC': (11, 0)
        }
        if str in months:
            return months[str]
        else:
            return None  # Return None for invalid input
        


