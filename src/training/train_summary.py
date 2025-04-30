import os
import numpy as np
import matplotlib.pyplot as plt


class TrainProcessSummary:

    def __init__(self, tr_losses, val_losses, tr_accs, val_accs, tr_f1s,
                 val_f1s, tr_specs, val_spec, tr_sens, val_sens, tr_times,
                 vl_times, data_name, save_path = "stats"):
        
        self.makedirs(save_path)
        self.xlbl, self.times_sv_name   = "Epochs", "times.png"
        self.data_name, self.save_path  = data_name, save_path
        self.tr_losses, self.val_losses = tr_losses, val_losses
        self.tr_accs,   self.val_accs   = tr_accs, val_accs
        self.tr_f1s,    self.val_f1s    = tr_f1s, val_f1s
        self.tr_specs,  self.val_spec   = tr_specs, val_spec
        self.tr_sens,   self.val_sens   = tr_sens, val_sens
        self.tr_times,  self.vl_times   = tr_times, vl_times
        self.get_ticks_labels(); self.learning_curves(); self.bar_plot()

        print(f"Learning curves can be found in {self.save_path}\n")
        print(f"Train times can be found in {self.save_path} directory under the name {self.times_sv_name}\n")
    
    def get_ticks_labels(self): self.xtics, self.xlabels = np.arange(len(self.tr_losses)), [i for i in range(1, len(self.tr_losses) + 1)]

    def makedirs(self, path): os.makedirs(path, exist_ok = True)
        
    def create_figure(self):    plt.figure(figsize = (10, 5))

    def move2cpu(self, data):   return [d.cpu() for d in data]

    def plot(self, data1, data2, plot_name, c1, c2):
        
        self.create_figure()
        need2bemoved = ["Sensitivity", "Specificity", "F1"]
        if plot_name in need2bemoved: data1 = self.move2cpu(data1); data2 = self.move2cpu(data2) 
        label = f"{plot_name} Scores"
        plt.plot(data1, label = f"Train {label}", color = c1); plt.plot(data2, label = f"Validation {label}", color = c2)
        plt.xlabel(self.xlbl); plt.ylabel(label); plt.title(f"Train and Validation {label}")
        plt.xticks(ticks = self.xtics, labels = self.xlabels); plt.legend(); plt.show()

    def save(self, save_name):

        sv_name = f"{self.data_name}_{save_name}"
        plt.savefig(f"{self.save_path}/{sv_name}")
        
    def learning_curves(self):
        
        self.plot(self.tr_losses, self.val_losses, "Loss",        "red",        "blue");         self.save("losses.png")
        self.plot(self.tr_accs,   self.val_accs,   "Accuracy",     "orangered",  "darkgreen");   self.save("accs.png")
        self.plot(self.tr_f1s,    self.val_f1s,    "F1",          "aquamarine", "greenyellow");  self.save("f1s.png")
        self.plot(self.tr_specs,  self.val_spec,   "Specificity", "violet",     "dodgerblue");   self.save("specs.png")        
        self.plot(self.tr_sens,   self.val_sens,   "Sensitivity", "gold",       "lightcoral");   self.save("sens.png")
        
    def bar_plot(self):
        
        self.create_figure();               
        
        plt.bar(self.xtics - 0.2, self.tr_times, 0.4, label = "Train") 
        plt.bar(self.xtics + 0.2, self.vl_times, 0.4, label = "Validation") 
        
        plt.xticks(self.xtics);   plt.xlabel(self.xlbl) 
        plt.ylabel("Seconds");    plt.title("Train and Validation Times") 
        plt.legend(); plt.show(); self.save(self.times_sv_name)
