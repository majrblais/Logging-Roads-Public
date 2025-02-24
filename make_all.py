#!/usr/bin/env python
# coding: utf-8

import os
import sys
import logging
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
from albumentations import Compose, Resize
from albumentations.pytorch import ToTensorV2
import tkinter as tk
from tkinter import filedialog, messagebox

# -------- Utility Functions --------
def list_files(directory):
    """List all files in a directory (sorted)."""
    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ])

def load_model(model_folder, model_type, encoder, device):
    """
    Loads a segmentation model from a given folder.
    Expects a single file ending with '_final.pth' in the folder.
    """
    model_files = [f for f in os.listdir(model_folder) if f.endswith("_final.pth")]
    if not model_files:
        logging.error(f"No model file found in {model_folder}")
        return None
    model_path = os.path.join(model_folder, model_files[0])
    ModelClass = getattr(smp, model_type, None)
    if ModelClass is None:
        logging.error(f"Model type {model_type} not recognized in segmentation_models_pytorch.")
        return None
    try:
        model = ModelClass(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None
        ).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        logging.info(f"Loaded model {model_type} with encoder {encoder} from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")
        return None

# -------- Image Processing Function --------
def process_images(models, input_folder, output_folder, transform, device, output_size):
    """
    Processes every image in the input folder:
      - Reads and converts the image to RGB.
      - Resizes the image to 512x512 (via the fixed transform) and converts it to a tensor.
      - Runs inference with each model.
      - Averages the binary predictions to create an ensemble mask.
      - Resizes the ensemble mask to the specified output dimensions.
      - Saves the output mask (as a grayscale image) to the output folder using the same filename.
    """
    image_paths = list_files(input_folder)
    logging.info(f"Found {len(image_paths)} images in {input_folder}")

    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logging.error(f"Error opening image {image_path}: {e}")
            continue

        # Convert the image to numpy array.
        image_np = np.array(image)
        augmented = transform(image=image_np)
        image_tensor = augmented['image'].to(device)

        predictions = []
        with torch.no_grad():
            for model_name, model in models.items():
                # Add batch dimension (1, C, H, W)
                input_tensor = image_tensor.unsqueeze(0).float()
                pred = model(input_tensor)
                pred = torch.sigmoid(pred)
                pred_bin = (pred > 0.5).float().cpu().numpy()[0]
                predictions.append(pred_bin)

        if predictions:
            # Average predictions and apply threshold to create binary ensemble mask.
            ensemble_mask = np.mean(predictions, axis=0)
            ensemble_mask = (ensemble_mask > 0.5).astype(np.uint8)
        else:
            logging.warning(f"No predictions for image {image_path}. Skipping.")
            continue

        # Create a PIL image from the ensemble mask (values in [0, 255]).
        mask_pil = Image.fromarray((ensemble_mask.squeeze() * 255).astype(np.uint8), mode="L")
        # Resize the mask to the user-specified output size.
        mask_pil = mask_pil.resize(output_size, resample=Image.NEAREST)

        output_path = os.path.join(output_folder, os.path.basename(image_path))
        try:
            mask_pil.save(output_path)
            logging.info(f"Saved ensemble mask to {output_path}")
        except Exception as e:
            logging.error(f"Error saving mask for image {image_path}: {e}")

# -------- Main Inference Function --------
def main_inference(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    # The transform now fixes the input size to 512x512.
    transform = Compose([
        Resize(512, 512),
        ToTensorV2()
    ])

    # Hardcoded models. The model folders are now located in "./weights"
    hardcoded_models = {
        "MAnet_resnet152": {
            "folder": os.path.join("./weights", "image_4000_resnet152_1e-05_MAnet_512_aug"),
            "model_type": "MAnet",
            "encoder": "resnet152"
        },
        "MAnet_densenet121": {
            "folder": os.path.join("./weights", "image_4000_densenet121_1e-05_MAnet_512_aug"),
            "model_type": "MAnet",
            "encoder": "densenet121"
        },
        "Unet_vgg16_bn": {
            "folder": os.path.join("./weights", "image_4000_vgg16_bn_1e-05_Unet_512_aug"),
            "model_type": "Unet",
            "encoder": "vgg16_bn"
        }
    }

    models = {}
    for model_key, info in hardcoded_models.items():
        folder = info["folder"]
        if not os.path.isdir(folder):
            logging.warning(f"Model folder {folder} does not exist. Skipping {model_key}.")
            continue
        model = load_model(folder, info["model_type"], info["encoder"], device)
        if model is not None:
            models[model_key] = model

    if not models:
        logging.error("No models loaded. Exiting.")
        return

    # Ensure the output folder exists.
    os.makedirs(config['output_folder'], exist_ok=True)

    # Process images from the input folder.
    output_size = (config['transform_width'], config['transform_height'])
    process_images(models, config['input_folder'], config['output_folder'], transform, device, output_size)

# -------- Configuration GUI --------
def open_config_window():
    config = {}
    root = tk.Tk()
    root.title("Ensemble Segmentation Inference Configuration")

    # Variables for configuration (model root is now fixed to "./weights")
    input_folder_var = tk.StringVar()
    output_folder_var = tk.StringVar()
    transform_width_var = tk.IntVar(value=512)
    transform_height_var = tk.IntVar(value=512)

    def browse_input_folder():
        selected = filedialog.askdirectory(title="Select Input Image Folder")
        if selected:
            input_folder_var.set(selected)

    def browse_output_folder():
        selected = filedialog.askdirectory(title="Select Output Folder")
        if selected:
            output_folder_var.set(selected)

    def submit():
        # Ensure all fields are filled.
        if not input_folder_var.get() or not output_folder_var.get():
            messagebox.showerror("Error", "Please select both input and output folders before launching inference.")
            return
        config['input_folder'] = input_folder_var.get()
        config['output_folder'] = output_folder_var.get()
        config['transform_width'] = transform_width_var.get()
        config['transform_height'] = transform_height_var.get()
        root.destroy()

    # Layout: Labels, Entries, and Buttons
    tk.Label(root, text="Input Folder:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    tk.Entry(root, textvariable=input_folder_var, width=50).grid(row=0, column=1, padx=5, pady=5)
    tk.Button(root, text="Browse", command=browse_input_folder).grid(row=0, column=2, padx=5, pady=5)

    tk.Label(root, text="Output Folder:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
    tk.Entry(root, textvariable=output_folder_var, width=50).grid(row=1, column=1, padx=5, pady=5)
    tk.Button(root, text="Browse", command=browse_output_folder).grid(row=1, column=2, padx=5, pady=5)

    tk.Label(root, text="Output Mask Width:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
    tk.Entry(root, textvariable=transform_width_var, width=10).grid(row=2, column=1, sticky="w", padx=5, pady=5)

    tk.Label(root, text="Output Mask Height:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
    tk.Entry(root, textvariable=transform_height_var, width=10).grid(row=3, column=1, sticky="w", padx=5, pady=5)

    tk.Button(root, text="Launch Inference", command=submit).grid(row=4, column=1, padx=5, pady=15)

    root.mainloop()
    return config

# -------- Main Entry Point --------
def main():
    # Set up logging.
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    config = open_config_window()
    main_inference(config)

if __name__ == '__main__':
    main()
