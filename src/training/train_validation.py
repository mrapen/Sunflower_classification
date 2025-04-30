import os
import datetime, torch, timm
from torchmetrics.classification import MulticlassStatScores, MulticlassF1Score
from tqdm import tqdm
from time import time


class TrainValidation:

    def __init__(self, model_name, tr_dl, val_dl, classes, device, lr, save_dir, 
                 run_name, data_name, epochs, bs, patience = 5, dev_mode = False):

        self.model_name, self.classes, self.device      = model_name, classes, device
        self.data_name, self.lr, self.save_dir, self.bs = data_name, lr, save_dir, bs
        self.tr_dl, self.val_dl, self.patience, self.dm = tr_dl, val_dl, patience, dev_mode
        self.run_name, self.epochs   = run_name, epochs
        self.run()
        
    def init_model(self): self.model = timm.create_model(self.model_name, pretrained = True, num_classes = len(self.classes))  
    
    def init_lists(self): self.tr_losses, self.val_losses, self.tr_sens, self.val_senss, self.tr_specs, self.val_specs, self.tr_accs, self.val_accs, self.tr_f1s, self.val_f1s, self.tr_times, self.vl_times = [], [], [], [], [], [], [], [], [], [], [], []

    def train_setup(self): 
                
        self.best_loss, self.threshold, self.not_improved = float(torch.inf), 0.01, 0
        self.stop_train, self.tr_len, self.val_len        = False, len(self.tr_dl), len(self.val_dl)
        self.cs_lbls = {"cos_pos": torch.tensor(1.).unsqueeze(0), "cos_neg": torch.tensor(-1.).unsqueeze(0)}
        self.ckpt_path = f"{self.save_dir}/{self.data_name}_{self.run_name}_{self.model_name}_best_model.pth"
        print(f"{self.run_name}_{self.model_name}_bs_{self.bs}")
        
        print(str(datetime.datetime.now()).split(".")[0])
        
        self.model.to(self.device).eval()
        self.ce_loss_fn   = torch.nn.CrossEntropyLoss() 
        self.cs_loss_fn   = torch.nn.CosineEmbeddingLoss(margin = 0.3) 
        self.optimizer   = torch.optim.Adam(params = self.model.parameters(), lr = self.lr)
        self.f1_score    = MulticlassF1Score(num_classes = len(self.classes), average = "micro").to(self.device) 
        self.stat_scores = MulticlassStatScores(num_classes = len(self.classes), average = "micro").to(self.device)
    
    def makedirs(self, path): os.makedirs(path, exist_ok = True)
    
    def get_fms(self, fms):
        
        """
        s
        This function gets feature map with size (bs, fm_shape, 7, 7)
        applies average pooling and returns feature map with shape (bs, fm_shape).
        
        Parameter:
        
            fm - feature map, tensor.
        
        Output:
        
            fm - reshaped feature map, tensor.
        
        """
        
        pool = torch.nn.AvgPool2d((fms[0].shape[2], fms[0].shape[3]))
        
        return [torch.reshape(pool(fm), (-1, fm.shape[1])) for fm in fms]
    
    def get_logits(self, ims): return [self.model.forward_features(im) for im in ims]

    def get_preds(self, fts): return [self.model.forward_head(ft) for ft in fts]
    
    def get_cs_loss(self, qry_fms, pos_fms, neg_fms): return self.cs_loss_fn(qry_fms, pos_fms, self.cs_lbls["cos_pos"].to(self.device)) + self.cs_loss_fn(qry_fms, neg_fms, self.cs_lbls["cos_neg"].to(self.device))

    def get_ce_loss(self, qry_preds, pos_preds, qry_lbls): return self.ce_loss_fn(qry_preds, qry_lbls) + self.ce_loss_fn(pos_preds, qry_lbls)
    
    def get_preds_loss(self, qry_ims, pos_ims, neg_ims, qry_lbls): 
        
        # Get logits
        qry_logits, pos_logits, neg_logits = self.get_logits([qry_ims, pos_ims, neg_ims])
        qry_preds, pos_preds = self.get_preds([qry_logits, pos_logits])
        
        # Contrastive loss
        qry_fms, pos_fms, neg_fms = self.get_fms([qry_logits, pos_logits, neg_logits])
        cs_loss = self.get_cs_loss(qry_fms, pos_fms, neg_fms)

        # Cross entropy loss
        ce_loss = self.get_ce_loss(qry_preds, pos_preds, qry_lbls)

        # Final loss
        loss = cs_loss + ce_loss
                
        return torch.argmax(qry_preds, dim = 1), loss

    def eval_train_batch(self, preds, gts, loss): 
        
        self.epoch_acc    += (preds == gts).sum().item()
        self.epoch_loss   += loss.item()
        self.epoch_f1     += self.f1_score(preds, gts)
        tp, fp, tn, fn, _  = self.stat_scores(preds, gts)
        self.spec         += tn / (tn + fp)
        self.sens         += tp / (tp + fn)

    def eval_valid_batch(self, preds, gts, loss): 

        self.val_epoch_loss   += loss.item()
        self.val_epoch_acc    += (preds == gts).sum().item()
        self.val_epoch_f1     +=  self.f1_score(preds, gts)
        tp, fp, tn, fn, _      = self.stat_scores(preds, gts)
        self.val_spec         += tn / (tn + fp)
        self.val_sens         += tp / (tp + fn)        
            
    def to_device(self, batch): return batch["qry_im"].to(self.device), batch["pos_im"].to(self.device), batch["neg_im"].to(self.device), batch["qry_gt"].to(self.device)
    
    def train_one_epoch(self, epoch):

        self.model.train()
        self.epoch_loss, self.epoch_acc, self.epoch_f1, self.loss, self.sens, self.spec = 0, 0, 0, 0, 0, 0

        tr_start = time(); 
        for idx, batch in tqdm(enumerate(self.tr_dl)):

            if self.dm:
                if idx == 1: break
            
            qry_ims, pos_ims, neg_ims, qry_im_lbls = self.to_device(batch)

            qry_preds, loss = self.get_preds_loss(qry_ims, pos_ims, neg_ims, qry_im_lbls)
            self.eval_train_batch(qry_preds, qry_im_lbls, loss)
            self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()

        tr_time = time() - tr_start; self.tr_times.append(tr_time)
        tr_loss_to_track = self.epoch_loss / self.tr_len
        tr_sens_to_track = self.sens / self.tr_len
        tr_spec_to_track = self.spec / self.tr_len
        tr_acc_to_track  = self.epoch_acc  / len(self.tr_dl.dataset)
        tr_f1_to_track   = self.epoch_f1   / self.tr_len
        self.tr_losses.append(tr_loss_to_track); self.tr_accs.append(tr_acc_to_track); self.tr_f1s.append(tr_f1_to_track); self.tr_sens.append(tr_sens_to_track); self.tr_specs.append(tr_spec_to_track)
        
        print("\n~~~~~~~~~~~~~~~~~~~~ TRAIN PROCESS STATS ~~~~~~~~~~~~~~~~~~~~")
        print(f"\n{epoch + 1}-epoch train process is completed!\n")
        print(f"{epoch + 1}-epoch train loss          -> {tr_loss_to_track:.3f}")
        print(f"{epoch + 1}-epoch train spec          -> {tr_spec_to_track:.3f}")
        print(f"{epoch + 1}-epoch train sens          -> {tr_sens_to_track:.3f}")
        print(f"{epoch + 1}-epoch train accuracy      -> {tr_acc_to_track:.3f}")
        print(f"{epoch + 1}-epoch train f1-score      -> {tr_f1_to_track:.3f}")

    def eval_one_epoch(self, epoch):

        self.val_epoch_loss, self.val_epoch_acc, self.val_epoch_f1, self.val_sens, self.val_spec = 0, 0, 0, 0, 0
        self.model.eval()
        with torch.no_grad():
            vl_start = time()
            for idx, batch in enumerate(self.val_dl):
                
                if self.dm:
                    if idx == 1: break
                
                qry_ims, pos_ims, neg_ims, qry_im_lbls = self.to_device(batch)

                qry_preds, loss = self.get_preds_loss(qry_ims, pos_ims, neg_ims, qry_im_lbls)
                self.eval_valid_batch(qry_preds, qry_im_lbls, loss)                           

            vl_time = time() - vl_start; self.vl_times.append(vl_time)
            val_loss_to_track = self.val_epoch_loss  / self.val_len
            val_sens_to_track = self.val_sens / self.val_len
            val_spec_to_track = self.val_spec / self.val_len
            val_acc_to_track  = self.val_epoch_acc   / len(self.val_dl.dataset)
            val_f1_to_track   = self.val_epoch_f1    / self.val_len
            self.val_losses.append(val_loss_to_track); self.val_accs.append(val_acc_to_track); self.val_f1s.append(val_f1_to_track); self.val_senss.append(val_sens_to_track); self.val_specs.append(val_spec_to_track)
            
            print(f"\n{epoch + 1}-epoch validation process is completed!\n")
            print(f"{epoch + 1}-epoch validation loss     -> {val_loss_to_track:.3f}")
            print(f"{epoch + 1}-epoch validation spec     -> {val_spec_to_track:.3f}")
            print(f"{epoch + 1}-epoch validation sens     -> {val_sens_to_track:.3f}")
            print(f"{epoch + 1}-epoch validation accuracy -> {val_acc_to_track:.3f}")
            print(f"{epoch + 1}-epoch validation f1-score -> {val_f1_to_track:.3f}")

            print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        return val_loss_to_track

    def save_best_model(self): torch.save(self.model.state_dict(), self.ckpt_path); print("Pretrained weights of the model with lowest loss are successfully saved!")
    
    def epoch_summary(self, metric):
        
        if (metric + self.threshold) < self.best_loss:
                
            print(f"\nValidation loss is decreased from {self.best_loss:.5f} to {metric:.5f}")
            print("Saving the best model with the lowest loss value...\n")
            self.best_loss = metric                
            self.save_best_model()            
            
        else:

            self.not_improved += 1
            print(f"\nValidation loss is not significantly decreased from {self.best_loss:.5f}. The current epoch loss is {metric:.5f}.")
            print(f"Validation loss value did not decrease for {self.not_improved} epochs")
            if self.not_improved == self.patience:
                print(f"Stop training since loss value did not decrease for {self.patience} epochs.")
                self.stop_train = True
    
    def train(self):
        
        print("Start training...")
        for epoch in range(self.epochs):
            if self.dm:
                if epoch == 1:
                    break
            self.train_one_epoch(epoch)
            loss = self.eval_one_epoch(epoch)
            self.epoch_summary(loss)
            if self.stop_train:
                break
    
    def get_stats(self): return [self.tr_losses, self.val_losses, self.tr_accs, self.val_accs, self.tr_f1s, self.val_f1s, self.tr_specs, self.val_specs, self.tr_sens, self.val_senss, self.tr_times, self.vl_times]

    def run(self): self.makedirs(self.save_dir); self.init_lists(); self.init_model(); self.train_setup(); self.train()
