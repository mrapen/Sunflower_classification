import torch
import os, timm
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from torchvision import transforms as T
from data.dataset import get_dls
from visualization.visualization import Visualization
from inference.inference import inference
from analysis.analysis import analyze_model_performance, analyze_class_performance


def main():
    # Display CUDA information
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # Settings
    root = "dataset"
    mean, std, size, bs = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224, 2
    tfs = T.Compose([T.ToTensor(), T.Resize(size=(size, size), antialias=False), T.Normalize(mean=mean, std=std)])

    # Create DataLoaders
    tr_dl, val_dl, ts_dl, classes, cls_counts = get_dls(root=root, transformations=tfs, bs=bs)
    print(f"Train dataloader length: {len(tr_dl)}")
    print(f"Validation dataloader length: {len(val_dl)}")
    print(f"Test dataloader length: {len(ts_dl)}")
    print(f"Classes: {classes}")

    # Define model parameters
    run_name, model_name = "triplet", "efficientnet_b0"
    save_dir, data_name, device = "saved_models", "stages", "cuda" if torch.cuda.is_available() else "cpu"
    
    # Path to the trained model 
    model_path = os.path.join(os.path.dirname(__file__), f"{save_dir}\{data_name}_{run_name}_{model_name}_best_model.pth")
    
    # Check if the trained model exists
    if os.path.exists(model_path):
        print(f"Found trained model at: {model_path}")
        
        # Data visualization (optional)
        vis = Visualization(vis_datas=[tr_dl, val_dl, ts_dl], n_ims=6, rows=2, 
                           cmap="rgb", cls_names=list(classes.keys()), cls_counts=cls_counts)
        vis.analysis()
        vis.visualization()

        # Load the model for inference
        model = timm.create_model(model_name=model_name, num_classes=len(classes))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Run inference with GradCAM visualization
        print("Running inference with the trained model...")
        inference(model=model, device=device, test_dl=ts_dl, num_ims=10, row=2, 
                 cls_names=list(classes.keys()), im_size=size)
        
        # Add analysis section here
        print("\n===== STARTING MODEL ANALYSIS =====")
        
        # 1. Calculate overall performance metrics
        all_preds, all_labels, accuracy = analyze_model_performance(model, ts_dl, device, classes)
        
        # 2. Analyze per-class performance
        analyze_class_performance(all_labels, all_preds, classes)
        
        print("===== MODEL ANALYSIS COMPLETE =====")
        
    else:
        print(f"Trained model not found at: {model_path}")
        print("You need to train the model first or provide the correct path to the trained model.")
        # Uncomment the following code if you want to train the model when not found
        
        from training.train_validation import TrainValidation
        from training.train_summary import TrainProcessSummary
        
        print("Training the model...")
        epochs = 20
        results = TrainValidation(model_name=model_name, tr_dl=tr_dl, val_dl=val_dl,
                                 classes=classes, device=device, lr=3e-4, save_dir=save_dir, 
                                 data_name=data_name, bs=bs, run_name=run_name, 
                                 epochs=epochs, patience=3, dev_mode=False).get_stats()
        
        # Analyze training
        TrainProcessSummary(*results, data_name=data_name)
        
        # Run inference with the newly trained model
        model = timm.create_model(model_name=model_name, num_classes=len(classes))
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
        inference(model=model, device=device, test_dl=ts_dl, num_ims=10, row=2, 
                 cls_names=list(classes.keys()), im_size=size)
        

if __name__ == '__main__':
    main()