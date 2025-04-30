import random
import os
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


class CustomDataset(Dataset):
    
    def __init__(self, root, cls_names = None, transformations = None):
   
        self.transformations = transformations
        data_dir = os.path.join(os.path.dirname(__file__), '..', root)
        self.imgs = np.load(os.path.join(data_dir, 'Sunflower_Stages.npy'))
        self.lbls = np.load(os.path.join(data_dir, 'Sunflower_Stages_Labels.npy'))
        self.cls_names, self.cls_counts, count = {} if not cls_names else cls_names, {}, 0
            
        for idx, (img, lbl) in enumerate(zip(self.imgs, self.lbls)):
            lbl = lbl[0]
            if self.cls_names is not None and lbl not in self.cls_names: self.cls_names[lbl] = count; count += 1
            if lbl not in self.cls_counts: self.cls_counts[lbl] = 1
            else: self.cls_counts[lbl] += 1
            
    def __len__(self): return len(self.imgs)

    def get_pos_neg_ims(self, qry_label):
        
        pos_im_paths = [im for (im, lbl) in zip(self.imgs, self.lbls) if qry_label == lbl]
        neg_im_paths = [im for (im, lbl) in zip(self.imgs, self.lbls) if qry_label != lbl]
        
        pos_rand_int = random.randint(a = 0, b = len(pos_im_paths) - 1)
        neg_rand_int = random.randint(a = 0, b = len(neg_im_paths) - 1)
        
        return pos_im_paths[pos_rand_int], neg_im_paths[neg_rand_int], [lbl for lbl in self.lbls if lbl != qry_label][0][0]

    def __getitem__(self, idx):
        
        qry_im = self.imgs[idx]
        qry_label = self.lbls[idx][0]

        pos_im, neg_im, neg_label = self.get_pos_neg_ims(qry_label = qry_label)

        qry_gt = self.cls_names[qry_label]
        neg_gt = self.cls_names[neg_label]

        if self.transformations is not None: qry_im = self.transformations(qry_im); pos_im = self.transformations(pos_im); neg_im = self.transformations(neg_im)

        data = {}

        data["qry_im"] = qry_im
        data["qry_gt"] = qry_gt
        data["pos_im"] = pos_im
        data["neg_im"] = neg_im
        data["neg_gt"] = neg_gt
            
        return data

def get_dls(root, transformations, bs, split = [0.9, 0.05, 0.05], ns = 0):
    
    ds = CustomDataset(root = root, transformations = transformations)
    cls_names = ds.cls_names; cls_counts = ds.cls_counts
    total_len = len(ds); tr_len = int(total_len * split[0]); vl_len = int(total_len * split[1]); ts_len = total_len - tr_len - vl_len
    tr_ds, vl_ds, ts_ds = random_split(dataset = ds, lengths = [tr_len, vl_len, ts_len])         

    tr_dl, val_dl, ts_dl = DataLoader(tr_ds, batch_size = bs, shuffle = True, num_workers = ns), DataLoader(vl_ds, batch_size = bs, shuffle = False, num_workers = ns), DataLoader(ts_ds, batch_size = 1, shuffle = False, num_workers = ns)
    
    return tr_dl, val_dl, ts_dl, cls_names, cls_counts