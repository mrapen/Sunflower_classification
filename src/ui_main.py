import os
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import timm
import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageTk
from torchvision import transforms
import json
from datetime import datetime
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image


class ImageClassifierInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Sunflower Stage Classifier")
        self.root.geometry("1200x800")
        
        # Model settings
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = {}
        self.image_size = 224
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        # Results tracking
        self.results_history = []
        self.current_image_path = None
        self.current_image_tensor = None
        
        # Create the UI
        self.create_ui()
        
        # Load model
        self.load_model_btn.invoke()
    
    def create_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create left panel (controls)
        left_panel = ttk.Frame(main_frame, padding="5", relief=tk.RAISED, borderwidth=1)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Create right panel (image display)
        right_panel = ttk.Frame(main_frame, padding="5")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add controls to left panel
        ttk.Label(left_panel, text="Model Controls", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Model loading
        model_frame = ttk.LabelFrame(left_panel, text="Model", padding="5")
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.model_path_var = tk.StringVar(value=os.path.join(os.path.dirname(__file__), "saved_models/stages_triplet_efficientnet_b0_best_model.pth"))
        ttk.Label(model_frame, text="Model Path:").pack(anchor=tk.W)
        model_path_entry = ttk.Entry(model_frame, textvariable=self.model_path_var, width=30)
        model_path_entry.pack(fill=tk.X, padx=5, pady=2)
        
        self.model_name_var = tk.StringVar(value="efficientnet_b0")
        ttk.Label(model_frame, text="Model Name:").pack(anchor=tk.W)
        model_name_entry = ttk.Entry(model_frame, textvariable=self.model_name_var, width=30)
        model_name_entry.pack(fill=tk.X, padx=5, pady=2)
        
        self.load_model_btn = ttk.Button(model_frame, text="Load Model", command=self.load_model)
        self.load_model_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Image loading
        image_frame = ttk.LabelFrame(left_panel, text="Image", padding="5")
        image_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.load_image_btn = ttk.Button(image_frame, text="Load Image", command=self.load_image)
        self.load_image_btn.pack(fill=tk.X, padx=5, pady=5)
        self.load_image_btn.config(state=tk.DISABLED)  # Disabled until model is loaded
        
        # Classification
        classify_frame = ttk.LabelFrame(left_panel, text="Classification", padding="5")
        classify_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.classify_btn = ttk.Button(classify_frame, text="Classify Image", command=self.classify_image)
        self.classify_btn.pack(fill=tk.X, padx=5, pady=5)
        self.classify_btn.config(state=tk.DISABLED)  # Disabled until image is loaded
        
        # Show GradCAM
        self.show_gradcam_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(classify_frame, text="Show GradCAM", variable=self.show_gradcam_var).pack(anchor=tk.W, padx=5)
        
        # Results
        results_frame = ttk.LabelFrame(left_panel, text="Results", padding="5")
        results_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.save_results_btn = ttk.Button(results_frame, text="Save Results", command=self.save_results)
        self.save_results_btn.pack(fill=tk.X, padx=5, pady=5)
        self.save_results_btn.config(state=tk.DISABLED)  # Disabled until results are available
        
        # Status information
        status_frame = ttk.LabelFrame(left_panel, text="Status", padding="5")
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor=tk.W, padx=5, pady=5)
        
        # Model information
        self.model_info_var = tk.StringVar(value="No model loaded")
        ttk.Label(status_frame, textvariable=self.model_info_var, wraplength=250).pack(anchor=tk.W, padx=5, pady=5)
        
        # Setup right panel with notebook for different views
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Image tab
        self.image_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.image_tab, text="Image")
        
        # Create image display
        self.image_canvas = tk.Canvas(self.image_tab, bg="white")
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # GradCAM tab
        self.gradcam_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.gradcam_tab, text="GradCAM")
        
        self.gradcam_canvas = tk.Canvas(self.gradcam_tab, bg="white")
        self.gradcam_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Results tab
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="Results")
        
        # Create results display (using matplotlib)
        self.results_figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.results_canvas = FigureCanvasTkAgg(self.results_figure, self.results_tab)
        self.results_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # History tab
        self.history_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.history_tab, text="History")
        
        # Create a treeview for history
        columns = ("Time", "Image", "Prediction", "Confidence")
        self.history_tree = ttk.Treeview(self.history_tab, columns=columns, show="headings")
        
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=100)
        
        self.history_tree.pack(fill=tk.BOTH, expand=True)
        
    def load_model(self):
        self.status_var.set("Loading model...")
        self.root.update()
        
        try:
            model_path = self.model_path_var.get()
            model_name = self.model_name_var.get()
            
            # Check if model file exists
            if not os.path.exists(model_path):
                self.status_var.set(f"Error: Model file not found at {model_path}")
                return
            
            # Try to load classes from a JSON file in the same directory
            classes_path = os.path.join(os.path.dirname(model_path), "classes.json")
            if os.path.exists(classes_path):
                with open(classes_path, 'r') as f:
                    self.classes = json.load(f)
            else:
                # Default classes if not found
                self.classes = {"Germination": 0, "Seedling": 1, "Vegetative": 2, "Bud Formation": 3, "Early Flowering": 4, "Full Flowering": 5, "Seed Development": 6, "Maturity": 7}
            
            num_classes = len(self.classes)
            
            # Create model
            self.model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
            
            # Load weights
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize GradCAM
            self.grad_cam = GradCAMPlusPlus(model=self.model, target_layers=[self.model.blocks[-1]])
            
            # Update status
            self.status_var.set("Model loaded successfully")
            self.model_info_var.set(f"Model: {model_name}\nClasses: {len(self.classes)}\nDevice: {self.device}")
            
            # Enable image loading
            self.load_image_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            self.status_var.set(f"Error loading model: {str(e)}")
            print(str(e))
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not file_path:
            return
        
        try:
            self.status_var.set(f"Loading image: {os.path.basename(file_path)}")
            self.root.update()
            
            # Load and preprocess the image
            self.current_image_path = file_path
            image = Image.open(file_path).convert('RGB')
            
            # Display original image
            self.display_image(image)
            
            # Preprocess for model
            self.current_image_tensor = self.transforms(image).unsqueeze(0).to(self.device)
            
            # Enable classification
            self.classify_btn.config(state=tk.NORMAL)
            self.status_var.set("Image loaded. Ready to classify.")
            
        except Exception as e:
            self.status_var.set(f"Error loading image: {str(e)}")
    
    def display_image(self, image):
        # Resize for display while maintaining aspect ratio
        display_height = 500
        ratio = display_height / image.height
        display_width = int(image.width * ratio)
        
        # Resize and convert to PhotoImage
        display_image = image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        photo_image = ImageTk.PhotoImage(display_image)
        
        # Update canvas
        self.image_canvas.config(width=display_width, height=display_height)
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=photo_image)
        self.image_canvas.image = photo_image  # Keep a reference
    
    def classify_image(self):
        if self.current_image_tensor is None or self.model is None:
            self.status_var.set("Error: No image or model loaded")
            return
        
        try:
            self.status_var.set("Classifying image...")
            self.root.update()
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(self.current_image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, prediction = torch.max(probabilities, dim=1)
            
            prediction = prediction.item()
            confidence = confidence.item()
            
            # Map prediction to class name
            class_names = list(self.classes.keys())
            prediction_name = class_names[prediction]
            
            # Update status
            self.status_var.set(f"Prediction: {prediction_name} (Confidence: {confidence:.2f})")
            
            # Create result visualization
            self.create_results_visualization(outputs.cpu(), prediction)
            
            # Create GradCAM visualization if enabled
            if self.show_gradcam_var.get():
                self.create_gradcam_visualization()
            
            # Add to history
            timestamp = datetime.now().strftime("%H:%M:%S")
            image_name = os.path.basename(self.current_image_path)
            self.results_history.append({
                "timestamp": timestamp,
                "image_path": self.current_image_path,
                "prediction": prediction_name,
                "confidence": confidence,
                "probabilities": probabilities[0].cpu().numpy().tolist()
            })
            
            # Update history view
            self.history_tree.insert("", "end", values=(
                timestamp, 
                image_name, 
                prediction_name, 
                f"{confidence:.2f}"
            ))
            
            # Enable save results
            self.save_results_btn.config(state=tk.NORMAL)
            
            # Switch to results tab
            self.notebook.select(2)  # Results tab
            
        except Exception as e:
            self.status_var.set(f"Error classifying image: {str(e)}")
    
    def create_results_visualization(self, outputs, prediction):
        # Get class names and probabilities
        class_names = list(self.classes.keys())
        probs = torch.nn.functional.softmax(outputs, dim=1)[0].numpy()
        
        # Clear previous plot
        self.results_figure.clf()
        ax = self.results_figure.add_subplot(111)
        
        # Create bar chart
        bars = ax.bar(range(len(class_names)), probs, color='lightgray')
        
        # Color the predicted class
        bars[prediction].set_color('green')
        
        # Add labels and styling
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_ylim([0, 1])
        ax.set_ylabel('Probability')
        ax.set_title('Classification Results')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add probability values above bars
        for i, prob in enumerate(probs):
            ax.text(i, prob + 0.05, f"{prob:.2f}", ha='center')
        
        self.results_figure.tight_layout()
        self.results_canvas.draw()
    
    def create_gradcam_visualization(self):
        # Generate GradCAM visualization
        grayscale_cam = self.grad_cam(input_tensor=self.current_image_tensor)
        grayscale_cam = grayscale_cam[0, :]
    
        # Завантаження зображення за допомогою PIL
        original_image = Image.open(self.current_image_path).convert('RGB')
        if original_image is None:
            raise ValueError(f"Не вдалося завантажити зображення: {self.current_image_path}")
        original_image = np.array(original_image)  # Конвертація в масив numpy
        original_image = cv2.resize(original_image, (self.image_size, self.image_size))
    
        # Normalize image
        normalized_image = original_image.astype(np.float32) / 255
    
        # Create visualization
        visualization = show_cam_on_image(normalized_image, grayscale_cam, use_rgb=True)
    
        # Convert to PIL Image and display
        gradcam_image = Image.fromarray(visualization)
        self.display_gradcam(gradcam_image)
    
        # Switch to GradCAM tab
        self.notebook.select(1)  # GradCAM tab
    
    def display_gradcam(self, image):
        # Resize for display while maintaining aspect ratio
        display_height = 500
        ratio = display_height / image.height
        display_width = int(image.width * ratio)
        
        # Resize and convert to PhotoImage
        display_image = image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        photo_image = ImageTk.PhotoImage(display_image)
        
        # Update canvas
        self.gradcam_canvas.config(width=display_width, height=display_height)
        self.gradcam_canvas.create_image(0, 0, anchor=tk.NW, image=photo_image)
        self.gradcam_canvas.image = photo_image  # Keep a reference
    
    def save_results(self):
        if not self.results_history:
            self.status_var.set("No results to save")
            return
        
        # Ask user for save location
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Convert numpy arrays to lists for JSON serialization
            results_to_save = []
            for result in self.results_history:
                result_copy = result.copy()
                results_to_save.append(result_copy)
            
            with open(file_path, 'w') as f:
                json.dump(results_to_save, f, indent=4)
            
            self.status_var.set(f"Results saved to {file_path}")
            
        except Exception as e:
            self.status_var.set(f"Error saving results: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierInterface(root)
    root.mainloop()