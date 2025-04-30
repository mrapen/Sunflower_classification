import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as T


class Visualization:

    def __init__(self, vis_datas, n_ims, rows, cmap = None, cls_names = None, cls_counts = None, t_type = "rgb"):

        self.n_ims, self.rows = n_ims, rows
        self.t_type, self.cmap,  = t_type, cmap
        self.cls_names = cls_names
        
        data_names = ["train", "val", "test"]
        self.vis_datas = {data_names[i]: vis_datas[i] for i in range(len(vis_datas))} 
        if isinstance(cls_counts, list): self.analysis_datas = {data_names[i]: cls_counts[i] for i in range(len(cls_counts))} 
        else: self.analysis_datas = {"all": cls_counts}

    def tn2np(self, t):
        
        gray_tfs = T.Compose([T.Normalize(mean = [ 0.], std = [1/0.5]), T.Normalize(mean = [-0.5], std = [1])])
        rgb_tfs = T.Compose([T.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]), T.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])])
        
        invTrans = gray_tfs if self.t_type == "gray" else rgb_tfs 
        
        return (invTrans(t) * 255).detach().squeeze().cpu().permute(1,2,0).numpy().astype(np.uint8) if self.t_type == "gray" else (invTrans(t) * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8)

    def plot(self, rows, cols, count, im, title = "Original Image"):
    
        plt.subplot(rows, cols, count)
        plt.imshow(self.tn2np(im))
        plt.axis("off"); plt.title(title)
        
        return count + 1

    def vis(self, data, save_name):

        print(f"{save_name.upper()} Data Visualization is in process...\n")
        assert self.cmap in ["rgb", "gray"], "Please choose rgb or gray cmap"
        if self.cmap == "rgb": cmap = "viridis"
        cols = self.n_ims // self.rows; count = 1
        
        plt.figure(figsize = (25, 20))
                
        indices = [np.random.randint(low = 0, high = len(data) - 1) for _ in range(self.n_ims)]

        for idx, index in enumerate(indices):
        
            if count == self.n_ims + 1: break
            
            meta_data = data[index]
            qry_im, pos_im, neg_im, qry_lbl, neg_lbl = meta_data["qry_im"], meta_data["pos_im"], meta_data["neg_im"], meta_data["qry_gt"], meta_data["neg_gt"]

            # First Plot
            count = self.plot(self.rows, cols, count, im = qry_im, title = f"Query Image \n Class -> {self.cls_names[qry_lbl]}")

            # Second Plot
            count = self.plot(self.rows, cols, count, im = pos_im, title = f"Positive Image \n Class -> {self.cls_names[qry_lbl]}")

            # Third Plot
            count = self.plot(self.rows, cols, count, im = neg_im, title = f"Negative Image \n Class -> {self.cls_names[neg_lbl]}")
        
        plt.show()

    def data_analysis(self, cls_counts, save_name):

        print("Data analysis is in process...\n")
        
        width, text_width, text_height = 0.7, 0.05, 2
        cls_names = list(cls_counts.keys()); counts = list(cls_counts.values())
        
        _, ax = plt.subplots(figsize = (20, 10))
        indices = np.arange(len(counts))

        ax.bar(indices, counts, width, color = "darkorange")
        ax.set_xlabel("Class Names", color = "black")
        ax.set_xticklabels(cls_names)
        ax.set(xticks = indices, xticklabels = cls_names)
        ax.set_ylabel("Data Counts", color = "black")
        ax.set_title(f"Dataset Class Imbalance Analysis")

        for i, v in enumerate(counts): ax.text(i - text_width, v + text_height, str(v), color = "royalblue")
    
    def visualization(self): [self.vis(data.dataset, save_name) for (save_name, data) in self.vis_datas.items()]
        
    def analysis(self): [self.data_analysis(data, save_name) for (save_name, data) in self.analysis_datas.items()]
 