import os, random, pickle
from os.path import join, isfile
from tqdm import tqdm
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms

meanstd = {
        'cifar10': [(0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)],
        'cifar100': [(0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)],
        'svhn': [(0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)]
        }

train_transform = {
        'cifar10': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*meanstd['cifar10'])
                ]),
        'cifar100': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*meanstd['cifar100'])
                ]),
        'svhn': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(*meanstd['svhn'])
                ])        
        }

train_kwargs = {
        'cifar10': {'train': True, 'download': True},
        'cifar100': {'train': True, 'download': True},
        'svhn': {'split': 'train', 'download': True}
        }

test_kwargs = {
        'cifar10': {'train': False, 'download': True},
        'cifar100': {'train': False, 'download': True},
        'svhn': {'split': 'test', 'download': True}
        }

def get_class_balanced_labels(targets, labels_per_class, save_path=None):
    num_classes = max(targets) + 1
    
    indices = list(range(len(targets)))
    random.shuffle(indices)
    
    label_count = {i: 0 for i in range(num_classes)}
    label_indices, unlabel_indices = [], []
    
    for idx in indices:
        if label_count[targets[idx]] < labels_per_class:
            label_indices.append(idx)
            label_count[targets[idx]] += 1
        else:
            unlabel_indices.append(idx)
    
    if save_path is not None:
        with open(join(save_path, 'label_indices.txt'), 'w') as f:
            for idx in label_indices:
                f.write(str(idx) + '\n')
        
    return label_indices, unlabel_indices

def get_repeated_indices(indices, num_iters, batch_size):
    length = num_iters * batch_size
    num_epochs = length // len(indices) + 1
    repeated_indices = []
    
    for epoch in tqdm(range(num_epochs), desc='Pre-allocating indices'):
        random.shuffle(indices)
        repeated_indices += indices
    
    return repeated_indices[:length]

class CIFAR10(dsets.CIFAR10):
    num_classes = 10
    def __init__(self, num_labels, num_iters, batch_size, return_unlabel=True, save_path=None, **kwargs):
        super(CIFAR10, self).__init__(**kwargs)
        labels_per_class = num_labels // self.num_classes
        self.return_unlabel = return_unlabel
        
        self.label_indices, self.unlabel_indices = get_class_balanced_labels(self.targets, labels_per_class, save_path)
        self.repeated_label_indices = get_repeated_indices(self.label_indices, num_iters, batch_size)
        if self.return_unlabel:
            self.repeated_unlabel_indices = get_repeated_indices(self.unlabel_indices, num_iters, batch_size)
        
    def __len__(self):
        return len(self.repeated_label_indices)
    
    def __getitem__(self, idx):
        label_idx = self.repeated_label_indices[idx]
        label_img, label_target = self.data[label_idx], self.targets[label_idx]
        label_img = Image.fromarray(label_img)

        if self.transform is not None:
            label_img = self.transform(label_img)
        if self.target_transform is not None:
            label_target = self.target_transform(label_target)
        
        if self.return_unlabel:
            unlabel_idx = self.repeated_unlabel_indices[idx]
            unlabel_img, unlabel_target = self.data[unlabel_idx], self.targets[unlabel_idx]
            unlabel_img = Image.fromarray(unlabel_img)
    
            if self.transform is not None:
                unlabel_img = self.transform(unlabel_img)
            if self.target_transform is not None:
                unlabel_target = self.target_transform(unlabel_target)
            return label_img, label_target, unlabel_img, unlabel_target
        else:
            return label_img, label_target

class CIFAR100(dsets.CIFAR100):
    num_classes = 100
    def __init__(self, num_labels, num_iters, batch_size, return_unlabel=True, additional='None', save_path=None, **kwargs):
        super(CIFAR100, self).__init__(**kwargs)
        labels_per_class = num_labels // self.num_classes
        self.return_unlabel = return_unlabel
        self.additional = additional
        
        if self.additional != 'None':
            with open(join(self.root, "tiny_237k.bin"), 'rb') as f:
                self.additional_data = pickle.load(f)
            with open(join(self.root, "tiny2cifar_labels.pkl"), 'rb') as f:
                self.additional_targets = pickle.load(f)
            
            self.repeated_label_indices = get_repeated_indices(list(range(len(self.data))), num_iters, batch_size)
            self.repeated_unlabel_indices = get_repeated_indices(list(range(len(self.additional_data))), num_iters, batch_size)
        else:
            self.label_indices, self.unlabel_indices = get_class_balanced_labels(self.targets, labels_per_class, save_path)
            self.repeated_label_indices = get_repeated_indices(self.label_indices, num_iters, batch_size)
            if self.return_unlabel:
                self.repeated_unlabel_indices = get_repeated_indices(self.unlabel_indices, num_iters, batch_size)
        
    def __len__(self):
        return len(self.repeated_label_indices)
    
    def __getitem__(self, idx):
        label_idx = self.repeated_label_indices[idx]
        label_img, label_target = self.data[label_idx], self.targets[label_idx]
        label_img = Image.fromarray(label_img)

        if self.transform is not None:
            label_img = self.transform(label_img)
        if self.target_transform is not None:
            label_target = self.target_transform(label_target)
        
        if self.additional != 'None':
            unlabel_idx = self.repeated_unlabel_indices[idx]
            unlabel_img = Image.fromarray(self.additional_data[unlabel_idx])
    
            if self.transform is not None:
                unlabel_img = self.transform(unlabel_img)
            unlabel_target = self.additional_targets[unlabel_idx]
            return label_img, label_target, unlabel_img, unlabel_target
        elif self.return_unlabel:
            unlabel_idx = self.repeated_unlabel_indices[idx]
            unlabel_img, unlabel_target = self.data[unlabel_idx], self.targets[unlabel_idx]
            unlabel_img = Image.fromarray(unlabel_img)
    
            if self.transform is not None:
                unlabel_img = self.transform(unlabel_img)
            if self.target_transform is not None:
                unlabel_target = self.target_transform(unlabel_target)
            return label_img, label_target, unlabel_img, unlabel_target
        else:
            return label_img, label_target

class SVHN(dsets.SVHN):
    num_classes = 10
    def __init__(self, num_labels, num_iters, batch_size, return_unlabel=True, save_path=None, **kwargs):
        super(SVHN, self).__init__(**kwargs)
        labels_per_class = num_labels // self.num_classes
        self.return_unlabel = return_unlabel
        
        self.label_indices, self.unlabel_indices = get_class_balanced_labels(self.labels, labels_per_class, save_path)
        self.repeated_label_indices = get_repeated_indices(self.label_indices, num_iters, batch_size)
        if self.return_unlabel:
            self.repeated_unlabel_indices = get_repeated_indices(self.unlabel_indices, num_iters, batch_size)
        
    def __len__(self):
        return len(self.repeated_label_indices)
    
    def __getitem__(self, idx):
        label_idx = self.repeated_label_indices[idx]
        label_img, label_target = self.data[label_idx], int(self.labels[label_idx])
        label_img = Image.fromarray(np.transpose(label_img, (1, 2, 0)))

        if self.transform is not None:
            label_img = self.transform(label_img)
        if self.target_transform is not None:
            label_target = self.target_transform(label_target)
        
        if self.return_unlabel:
            unlabel_idx = self.repeated_unlabel_indices[idx]
            unlabel_img, unlabel_target = self.data[unlabel_idx], int(self.labels[unlabel_idx])
            unlabel_img = Image.fromarray(np.transpose(unlabel_img, (1, 2, 0)))
    
            if self.transform is not None:
                unlabel_img = self.transform(unlabel_img)
            if self.target_transform is not None:
                unlabel_target = self.target_transform(unlabel_target)
            return label_img, label_target, unlabel_img, unlabel_target
        else:
            return label_img, label_target

train_dset = {
        'cifar10': CIFAR10,
        'cifar100': CIFAR100,
        'svhn': SVHN
        }

test_dset = {
        'cifar10': dsets.CIFAR10,
        'cifar100': dsets.CIFAR100,
        'svhn': dsets.SVHN
        }

def dataloader(dset, path, bs, num_workers, num_labels, num_iters, return_unlabel=True, additional='None', save_path=None):
    assert dset in ["cifar10", "cifar100", "svhn"]
    assert additional in ['None', '237k', '500k']
    if additional != 'None':
        assert dset == "cifar100" and num_labels == 50000, 'Use additional data only for cifar100 dataset with 50k labeled data'
        train_kwargs[dset]["additional"] = additional
    
    train_dataset = train_dset[dset](
            root = path,
            num_labels = num_labels,
            num_iters = num_iters,
            batch_size = bs,
            return_unlabel = return_unlabel,
            transform = train_transform[dset],
            save_path = save_path,
            **train_kwargs[dset]
            )
    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=num_workers, shuffle=False)
    
    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*meanstd[dset])
            ])
    test_dataset = test_dset[dset](root=path, transform=test_transform, **test_kwargs[dset])
    test_loader = DataLoader(test_dataset, batch_size=100, num_workers=num_workers, shuffle=False)
    
    return iter(train_loader), test_loader

"""
class TinyImages(Dataset):
    ''' Tiny Images Dataset '''
    def __init__(self, root='data', which=None, transform=None, NO_LABEL=-1):
        self.data_path = join(root, 'tiny_images.bin')
        pkl_path= join(root, 'tiny_index.pkl')
        meta_path = join(root, 'cifar100_meta.meta')
        with open(pkl_path, 'rb') as f:
            tinyimg_index = pickle.load(f)
        
        if which == '237k':
            print("Using all classes common with CIFAR-100.")
            with open(meta_path, 'rb') as f:
                cifar_labels = pickle.load(f)['fine_label_names']
            cifar_to_tinyimg = { 'maple_tree': 'maple', 'aquarium_fish' : 'fish' }
            cifar_labels = [l if l not in cifar_to_tinyimg else cifar_to_tinyimg[l] for l in cifar_labels]
            load_indices = sum([list(range(*tinyimg_index[label])) for label in cifar_labels], [])
        
        elif which == '500k':
            print("Using {} random images.".format(which))
            idxs_filename = join(root, 'tiny_idxs500k.pkl')
            load_indices = pickle.load(open(idxs_filename, 'rb'))

        # now we have load_indices
        self.indices = load_indices
        self.len = len(self.indices)
        self.transform = transform
        self.no_label = NO_LABEL
        print("Length of the dataset = {}".format(self.len))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img = self.load_tiny_image(self.indices[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.no_label

    def load_tiny_image(self, idx):
        with open(self.data_path, 'rb') as f:
            f.seek(3072 * idx)
            img = np.fromfile(f, dtype='uint8', count=3072).reshape(3, 32, 32).transpose((0, 2, 1))
            img = Image.fromarray(np.rollaxis(img, 0, 3))
        return img
"""

class TinyImages(Dataset):
    """ Tiny Images Dataset """
    def __init__(self, root='data', which=None, transform=None, NO_LABEL=-1):
        with open(join(root, "tiny_237k.bin"), 'rb') as f:
            self.data = pickle.load(f)
        with open(join(root, "tiny2cifar_labels.pkl"), 'rb') as f:
            self.targets = pickle.load(f)
        self.transform = transform
        # self.no_label = NO_LABEL

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = Image.fromarray(self.data[index])
        targets = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, targets

    def __len__(self):
        return len(self.data)

