# Deep Learning Mask Extraction Workflow

This repository contains three Python code modules that implement a four-step workflow for extracting training data and generating segmentation masks using ArcGIS and deep learning. The workflow is divided into the following steps:

---

## Workflow Overview

1. **ArcGIS Fetching (RGB & DSM Extraction)**  
   This code module uses ArcGIS Pro’s Python API (ArcPy) with the Image Analyst extension to capture both RGB and DSM imagery from a specified area.  
   - **DSM Limitations:** DSM services often have a size limit (e.g., 4000×4000 cells at a given cell size). To work around this, the code splits a larger area into smaller tiles.  
   - **Feature Class:** The feature class (e.g., a road map) is intentionally left empty. This allows you to start from scratch and process new regions without any predefined features.  
   - **Note:** If you only require RGB images, you can use the native ArcGIS tool directly without this custom Python code.
   - **Note:** If the selected region is not a multiple of the tile size (e.g 4000x4000), some images will have black regions. (Todo, fix)

2. **Environment Setup**  
   The deep learning masking and mask modification code modules are designed to run from a Python environment (for example, using an Anaconda prompt).  
   - **Creating the Environment:**  
     Create and activate a new Conda environment, then install the required libraries:
     ```bash
     conda create --name deepmask python=3.8 -y
     conda activate deepmask
     pip install torch segmentation-models-pytorch albumentations pillow
     ```
   - This environment provides all necessary libraries such as **torch**, **segmentation_models_pytorch**, **albumentations**, and **PIL** (Pillow).

3. **Deep Learning Masking**  
   The second code module uses deep learning models to generate segmentation masks:
   - **Model Loading:** Pre-trained segmentation models are loaded from a hardcoded `./weights` folder using the `segmentation_models_pytorch` library.
   - **Preprocessing:** Each input image is resized to 512×512 pixels using Albumentations before inference.
   - **Ensembling:** Predictions from multiple models are averaged to produce a binary segmentation mask.
   - **Postprocessing:** The resulting ensemble mask is resized to user-specified dimensions.
   - **Usage:** Run this module from your Python environment (e.g., via `python production_code_gui.py`).

4. **Mask Modifying**  
   The third code module modifies the export process by extending the overall extent from the input layer so that its width and height become multiples of the tile size:
   - **Extent Adjustment:** If the original extent (e.g., from the `"test"` layer) is not a multiple of the tile size, the code extends the extent (symmetrically or otherwise) to meet this requirement. This helps ensure that each exported tile is uniformly sized (e.g., 4000×4000 pixels) and minimizes issues like black corners.  
   - **Usage:** Run this module (e.g., via `python modify_mask_gui.py`) to generate uniformly sized output tiles.

---

## Detailed Steps

### Step 1: ArcGIS Fetching
- **Purpose:**  
  Select an area and capture both RGB and DSM images.
- **Key Points:**  
  - Uses `ExportTrainingDataForDeepLearning` from ArcPy.
  - Splits the area into smaller tiles due to DSM size limitations.
  - Feature class is left empty to allow processing of new regions.
- **When to Use:**  
  Use this module if you need DSM data alongside RGB imagery. For RGB-only extraction, the built-in ArcGIS tool may be sufficient.

### Step 2: Environment Setup
- **Purpose:**  
  Prepare the Python environment for deep learning processing.
- **Instructions:**  
  1. Open an Anaconda prompt.
  2. Create a new environment and install required libraries:
     ```bash
     conda create --name deepmask python=3.8 -y
     conda activate deepmask
     pip install torch segmentation-models-pytorch albumentations pillow
     ```
  3. Ensure you use the Python interpreter that has access to the ArcGIS Pro libraries if needed.

### Step 3: Deep Learning Masking
- **Purpose:**  
  Generate segmentation masks using an ensemble of deep learning models.
- **Key Points:**  
  - Input images are preprocessed (resized to 512×512).
  - Multiple models are used, and their outputs are averaged to create a final binary mask.
  - The output mask is resized to the dimensions specified by the user.
- **Usage:**  
  Run this module with the command (e.g., `python production_code_gui.py`).

### Step 4: Mask Modifying
- **Purpose:**  
  Adjust the overall extent so that it is a multiple of the tile size, ensuring uniformly sized exported tiles.
- **Key Points:**  
  - Checks the original extent of the region (from the `"test"` layer).
  - If the extent isn’t an exact multiple of the tile size (e.g., 4000 map units), the code extends the extent to the next multiple.
  - This helps prevent black areas in exported tiles.
- **Usage:**  
  Run this module with the command (e.g., `python modify_mask_gui.py`).

---

## How to Run

1. **ArcGIS Fetching (Step 1):**  
   Run the ArcGIS code from within ArcGIS Pro’s Python environment if you need to capture both RGB and DSM images.

2. **Environment Setup (Step 2):**  
   Follow the instructions above to create and activate the Conda environment.

3. **Deep Learning Masking (Step 3):**  
   Run the deep learning masking code (e.g., `python production_code_gui.py`) to generate segmentation masks from your images.

4. **Mask Modifying (Step 4):**  
   Run the mask modifying code (e.g., `python modify_mask_gui.py`) to adjust the extent and ensure each exported tile is uniformly sized.

---

## Additional Notes

- **DSM Limitations:**  
  The DSM service (e.g., `"LidarProducts/DSM_HS"`) has size restrictions. The ArcGIS fetching code splits the area into tiles to work within these limits.
- **Modularity:**  
  Each code module is independent and can be run separately depending on your requirements.
- **Debugging:**  
  Detailed debug output is printed to the console to help diagnose any issues during processing.

---

Update or modify this README as your workflow evolves.
