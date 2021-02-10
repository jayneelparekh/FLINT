import torch

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
#from torchvision import transforms
import pickle
import numpy as np
from PIL import Image
import torch

#from libcpab.libcpab.pytorch import cpab


class QuickDraw(Dataset):
 
    def __init__(self, ncat=10, mode='train', root_dir=None):

        self.mode = mode
        self.root_dir = root_dir # root_dir should be the one within which the '.npy' are present, categories.txt should also be present

        cat_file = open(self.root_dir + 'categories.txt', 'r')                                                                                             
        self.cat = cat_file.readlines()                                                                                                                                           
        self.cat = [x.rstrip('\n') for x in self.cat]
        if ncat == 10:
            self.cat = ['apple', 'banana', 'carrot', 'grapes', 'ant', 'cat', 'dog', 'cow', 'lion', 'frog']
            self.cat.sort()
            print ('Selected categories', self.cat) 
        elif ncat == 20:
            self.cat = ['apple', 'banana', 'carrot', 'grapes', 'ant', 'cat', 'dog', 'cow', 'lion', 'frog', 'camel', 'airplane', 'broccoli', 'bus', 'butterfly', 'cactus', 'camera', 'calculator', 'alarm clock', 'ambulance']
            self.cat.sort()
            print ('Selected categories', self.cat)              
        # Organize categories
        # --- Write code here ---
        
        self.dict_name = 'info_' + str(len(self.cat)) + '.pkl'
        try:
            pkl_file = open(self.root_dir + self.dict_name, 'rb')
        except:
            self.build_info_dict()

        pkl_file = open(self.root_dir + self.dict_name, 'rb')
        self.info = pickle.load(pkl_file)
        pkl_file.close()
        


    def build_info_dict(self):
        # This is a useless function now
        print ('Building the info dictionary. selecting images for training and testing from organized categories')
        info_dict = {}
        info_dict['train'] = {}
        info_dict['test'] = {}
        # Select images for training, testing
        # --- Code here ---
        for i in range(len(self.cat)):
            cur_cat = self.cat[i]
            cur_samp = np.load(self.root_dir + cur_cat + '_small.npy')
            n_samp = cur_samp.shape[0]
            idx = np.arange(0, n_samp)
            np.random.shuffle(idx)
            info_dict['train'][cur_cat] = idx[:8000]
            info_dict['test'][cur_cat] = idx[8000:]
        
        info_dict['train']['samp_per_class'] = 8000
        info_dict['test']['samp_per_class'] = 2000             
        f = open(self.root_dir + self.dict_name, 'wb')
        pickle.dump(info_dict, f)
        f.close()
        return

    def set_mode(self, new_mode):
        self.mode = new_mode

    def __len__(self):
        return len(self.cat) * self.info[self.mode]['samp_per_class']

    def __getitem__(self, idx):
        samp_per_class = self.info[self.mode]['samp_per_class']
        y = idx // samp_per_class
        cat = self.cat[y]
        class_idx = idx % samp_per_class
        if self.mode == 'test':
            class_idx = self.info['train']['samp_per_class'] + class_idx # Ensure picking from samples not from training data
        arr = np.load(self.root_dir + cat + '_small.npy')
        image = arr[class_idx].reshape([28, 28])
        image = torch.as_tensor(image).float() / 255.0
        return image.unsqueeze(0), torch.as_tensor(y)#, torch.as_tensor(vis_locs).float()



        
        
        
