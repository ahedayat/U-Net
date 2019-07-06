import random
import torch 

import numpy as np
import torchvision.transforms as transforms 

from PIL import Image
from torch.utils.data import DataLoader, Dataset

class CamVidLoader(Dataset):
    def __init__(self, root_path, anns_dir_name=None, ann_version=1, data_transform=None, target_transform=None):
        # self.data_path = root_path + '/processed_images'
        self.data_path = '{}/images'.format(root_path)
        if anns_dir_name is None:
            self.anns_path = root_path + '/annotations_v{}'.format(ann_version)
        else:
            self.anns_path = '{}/{}'.format(root_path, anns_dir_name)
        self.file_names = [line.rstrip('\n') for line in open(root_path+'/file_names.txt')]
        self.data_transform = data_transform
        self.target_transform = target_transform
        self.labels = list()
        self.colors = list()
        for line in open('{}/../labels_v{}.txt'.format(root_path,ann_version)):
            r, g, b, label = line.split()
            r, g, b = int(r), int(g), int(b)
            self.labels.append(label)
            self.colors.append( (r,g,b) )

        # random.shuffle(self.file_names)

    def __getitem__(self,ix):
        file_name = self.file_names[ix]
        data = Image.open( self.data_path + '/' + file_name )
        # data = transforms.ToTensor()(data)
        ann = Image.open( self.anns_path + '/' + file_name )
        # print('&&&& data : {}'.format(self.data_path + '/' + file_name))
        # print('&&&& ann : {}'.format(self.anns_path + '/' + file_name))

        if self.data_transform is not None:
            data = self.data_transform(data)

        if self.target_transform is not None:
            ann = self.target_transform(ann)

        data = np.array(data)
        ann = np.array(ann)

        weidth, height = ann.shape

        mask = np.zeros( (len(self.labels), weidth, height) )
        for label_ix, label in enumerate(self.labels):
            label_mask = ann==label_ix
            mask[label_ix,label_mask] = 1

        data = torch.tensor(data, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.float)
        ann = torch.tensor(ann, dtype=torch.int64)

        return data, mask, ann

    def __len__(self):
        return len(self.file_names)
    
    def get_labels(self):
        return self.labels, self.colors

    def get_label(self,id):
        return self.labels[id], self.colors[id]

    def num_labels(self):
        return len(self.labels)