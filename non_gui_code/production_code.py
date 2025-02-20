#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import logging
import argparse
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
from albumentations import Compose, Resize
from albumentations.pytorch import ToTensorV2

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
def process_images(models, input_folder, output_folder, transform, device):
    """
    Processes every image in the input folder:
      - Reads and converts the image to RGB.
      - Applies a resize+tensor transform.
      - Runs inference with each model.
      - Averages the binary predictions to create an ensemble mask.
      - Resizes the ensemble mask back to the original image dimensions.
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

        original_size = image.size  # (width, height)
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
            # Average predictions and apply threshold to create binary ensemble mask
            ensemble_mask = np.mean(predictions, axis=0)
            ensemble_mask = (ensemble_mask > 0.5).astype(np.uint8)
        else:
            logging.warning(f"No predictions for image {image_path}. Skipping.")
            continue

        # Resize ensemble mask back to original dimensions (using nearest neighbor interpolation)
        mask_pil = Image.fromarray((ensemble_mask.squeeze() * 255).astype(np.uint8), mode="L")
        mask_pil = mask_pil.resize(original_size, resample=Image.NEAREST)

        output_path = os.path.join(output_folder, os.path.basename(image_path))
        try:
            mask_pil.save(output_path)
            logging.info(f"Saved ensemble mask to {output_path}")
        except Exception as e:
            logging.error(f"Error saving mask for image {image_path}: {e}")

# -------- Main Function --------
def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    # Define transformation: resize to 512x512 and convert to tensor.
    transform = Compose([
        Resize(512, 512),
        ToTensorV2()
    ])

    # Hardcoded models
    # Each entry contains the folder name (relative to model_root), the model type, and the encoder.
    hardcoded_models = {
        "MAnet_resnet152": {
            "folder": os.path.join(args.model_root, "image_4000_resnet152_1e-05_MAnet_512_aug"),
            "model_type": "MAnet",
            "encoder": "resnet152"
        },
        "MAnet_densenet121": {
            "folder": os.path.join(args.model_root, "image_4000_densenet121_1e-05_MAnet_512_aug"),
            "model_type": "MAnet",
            "encoder": "densenet121"
        },
        "Unet_vgg16_bn": {
            "folder": os.path.join(args.model_root, "image_4000_vgg16_bn_1e-05_Unet_512_aug"),
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
    os.makedirs(args.output_folder, exist_ok=True)

    # Process images from the input folder.
    process_images(models, args.input_folder, args.output_folder, transform, device)

# -------- Entry Point --------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Ensemble Segmentation Inference Script (Hardcoded Models)"
    )
    parser.add_argument("--model_root", type=str, default="./",
                        help="Root directory containing the model folders")
    parser.add_argument("--input_folder", type=str, default="./",
                        help="Folder containing input images to process")
    parser.add_argument("--output_folder", type=str, default="./ensemble_predictions",
                        help="Folder to save ensemble mask outputs")
    args = parser.parse_args()

    # Set up logging.
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    main(args)


# In[ ]:




