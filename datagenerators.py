import numpy as np
import sys
from torchvision import transforms
from torch.utils.data import Dataset
import nibabel as nib

def load_example_by_name(vol_name, seg_name=None):
    """
    load a specific volume and segmentation
    """
    X = nib.load(vol_name).get_data()
    X = np.reshape(X, (1,) + X.shape + (1,))

    return_vals = [X]

    if(seg_name):
        X_seg = nib.load(seg_name).get_data()
        X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))
        return_vals.append(X_seg)

    return tuple(return_vals)


def load_volfile(datafile):
    """
    load volume file
    formats: nii, nii.gz, mgz, npz
    if it's a npz (compressed numpy), assume variable names 'vol_data'
    """
    assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file: %s' % datafile

    if datafile.endswith(('.nii', '.nii.gz', '.mgz')):
        # import nibabel
        if 'nib' not in sys.modules:
            try:
                import nibabel as nib
            except:
                print('Failed to import nibabel. need nibabel library for these data file types.')

        X = nib.load(datafile).get_data()

    else:  # npz
        X = np.load(datafile)['vol_data']

    return X


class MRIDataset(Dataset):
    def __init__(self, train_vol_names, atlas_file):
        self.train_vol_names = train_vol_names
        self.len = len(self.train_vol_names)
        self.atlas_file = atlas_file

        '''class torchvision.transforms.ToTensor'''
        self.toTensor = transforms.ToTensor()

    def __getitem__(self, i):
        atlas = np.load(self.atlas_file)['vol']
        atlas = self.toTensor(atlas)
        atlas = atlas[np.newaxis, ...]
        atlas = atlas.permute(0, 2, 3, 1).float()
        index = i % self.len
        X = load_volfile(self.train_vol_names[index])
        X = self.toTensor(X)
        X = X[np.newaxis, ...]
        X = X.permute(0, 2, 3, 1).float()

        return X, atlas

    def __len__(self):
        return len(self.train_vol_names)


class T1T2Dataset(Dataset):
    def __init__(self, train_vol_names, atlas_file):
        self.train_vol_names = train_vol_names
        self.len = len(self.train_vol_names)
        self.atlas_file = atlas_file

        '''class torchvision.transforms.ToTensor'''
        self.toTensor = transforms.ToTensor()

    def __getitem__(self, i):
        atlas = load_volfile(self.atlas_file)
        atlas = self.toTensor(atlas)
        atlas = atlas[np.newaxis, ...]
        atlas = atlas.permute(0, 2, 3, 1).float()
        index = i % self.len
        X = load_volfile(self.train_vol_names[index])
        X = self.toTensor(X)
        X = X[np.newaxis, ...]
        X = X.permute(0, 2, 3, 1).float()

        return X, atlas

    def __len__(self):
        return len(self.train_vol_names)