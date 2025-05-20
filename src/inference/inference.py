import torch
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as T
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm


def tensor_2_im(t, t_type = "rgb"):
    
    gray_tfs = T.Compose([T.Normalize(mean = [ 0.], std = [1/0.5]), T.Normalize(mean = [-0.5], std = [1])])
    rgb_tfs = T.Compose([T.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]), T.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])])
    
    inv_trans = gray_tfs if t_type == "gray" else rgb_tfs
    
    return (inv_trans(t) * 255).detach().squeeze().cpu().permute(1,2,0).numpy().astype(np.uint8) if t_type == "gray" else (inv_trans(t) * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8)

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
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    with torch.no_grad():
        preds, images, lbls, logitss = [], [], [], []
        acc, count                   = 0, 1
        for idx, batch in tqdm(enumerate(test_dl)):
            # if idx == num_ims: break
            image, ground_truth = batch["qry_im"].to(device), batch["qry_gt"].to(device)
            features = model.forward_features(image)
            logits = model.forward_head(features)
            pred_class = torch.argmax(logits, dim = 1)
            acc += (pred_class == ground_truth).sum().item()
            images.append(image)
            logitss.append(logits)
            preds.append(pred_class)
            lbls.append(ground_truth.item())
        dataset_size = len(test_dl.dataset) or len(images)
        print(f"Accuracy of the model on the test data -> {(acc / dataset_size):.3f}")
    plt.figure(figsize = (20, 10))
    cam = GradCAMPlusPlus(model=model, target_layers=[model.blocks[-1]])
    cam.device = device
    if len(images) == 0:
        print("No image to display; skipping rendering.")
    else:
        indekslar = [random.randint(0, len(images) - 1) for _ in range(num_ims)]
        for _, index in enumerate(indekslar):
        
            image = tensor_2_im(images[index].squeeze())
            pred = preds[index]
                
            # Start plot
            plt.subplot(row, 2 * num_ims // row, count); count += 1
            plt.imshow(image, cmap = "gray"); plt.axis("off")
        
            targets = [ClassifierOutputTarget(int(preds[index]))]
            grayscale_cam = cam(input_tensor=images[index], targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            resized_cam = cv2.resize(grayscale_cam, (im_size, im_size), interpolation=cv2.INTER_LINEAR)
            visualization = show_cam_on_image(image / 255, resized_cam, image_weight=0.4, use_rgb = True)
            plt.imshow(cv2.resize(visualization, (im_size, im_size), interpolation=cv2.INTER_LINEAR), alpha=0.7, cmap='jet'); plt.axis("off")
            plt.subplot(row, 2 * num_ims // row, count); count += 1
            plot_value_array(logits = logitss[index], gt = lbls[index], cls_names = cls_names)
        
            if cls_names is not None:
                plt.title(f"GT -> {cls_names[int(lbls[index])]} ; PRED -> {cls_names[int(preds[index])]}", color=("green" if {cls_names[int(lbls[index])]} == {cls_names[int(preds[index])]} else "red"))
            else:
                plt.title(f"GT -> {ground_truth} ; PRED -> {pred}")
    plt.show()
