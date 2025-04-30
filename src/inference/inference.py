import torch
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as T
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm


def tensor_2_im(t, t_type = "rgb"):
    
    gray_tfs = T.Compose([T.Normalize(mean = [ 0.], std = [1/0.5]), T.Normalize(mean = [-0.5], std = [1])])
    rgb_tfs = T.Compose([T.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]), T.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])])
    
    invTrans = gray_tfs if t_type == "gray" else rgb_tfs 
    
    return (invTrans(t) * 255).detach().squeeze().cpu().permute(1,2,0).numpy().astype(np.uint8) if t_type == "gray" else (invTrans(t) * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8)

def plot_value_array(logits, gt, cls_names):
    
    probs = torch.nn.functional.softmax(logits, dim = 1)
    pred_score, pred_class = torch.max(probs, dim = 1) 

    plt.grid(visible = True)
    plt.xticks(range(len(cls_names)), cls_names, rotation='vertical')
    plt.yticks(np.arange(start = 0., stop = 1.01, step = 0.1))
    bar = plt.bar(range(len(cls_names)), [p.item() for p in probs[0]], color="#777777")
    plt.ylim([0, 1])   
    # Color the bars based on the criteria
    if pred_class.item() == gt:
        bar[pred_class].set_color('green')
    else:
        bar[pred_class].set_color('red')

def inference(model, device, test_dl, num_ims, row, cls_names = None, im_size = 224):
    
    with torch.no_grad():
        preds, images, lbls, logitss = [], [], [], []
        acc, count                   = 0, 1
        for idx, batch in tqdm(enumerate(test_dl)):
            # if idx == num_ims: break
            im, gt = batch["qry_im"].to(device), batch["qry_gt"].to(device)
            logits  = model(im)
            pred_class   = torch.argmax(logits, dim = 1)
            acc += (pred_class == gt).sum().item()
            images.append(im)
            logitss.append(logits)
            preds.append(pred_class)
            lbls.append(gt.item())
        
        print(f"Accuracy of the model on the test data -> {(acc / len(test_dl.dataset)):.3f}")
        
    plt.figure(figsize = (20, 10))
    cam = GradCAMPlusPlus(model=model, target_layers=[model.blocks[-1]])
    indekslar = [random.randint(0, len(images) - 1) for _ in range(num_ims)]
    for idx, indeks in enumerate(indekslar):
        
        im = tensor_2_im(images[indeks].squeeze())
        pred_idx = preds[indeks]
                
        # Start plot
        plt.subplot(row, 2 * num_ims // row, count); count += 1
        plt.imshow(im, cmap = "gray"); plt.axis("off")
        
        grayscale_cam = cam(input_tensor=images[indeks])
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(im / 255, grayscale_cam, image_weight=0.4, use_rgb = True)
        plt.imshow(cv2.resize(visualization, (im_size, im_size), interpolation=cv2.INTER_LINEAR), alpha=0.7, cmap='jet'); plt.axis("off")
        plt.subplot(row, 2 * num_ims // row, count); count += 1
        plot_value_array(logits = logitss[indeks], gt = lbls[indeks], cls_names = cls_names)
        
        if cls_names is not None: plt.title(f"GT -> {cls_names[int(lbls[indeks])]} ; PRED -> {cls_names[int(preds[indeks])]}", color=("green" if {cls_names[int(lbls[indeks])]} == {cls_names[int(preds[indeks])]} else "red"))
        else: plt.title(f"GT -> {gt} ; PRED -> {pred}") 
