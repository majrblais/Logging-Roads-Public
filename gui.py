#!/usr/bin/env python3
import cv2
import os
import sys
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import logging

# --- Additional imports for model inference ---
import torch
import segmentation_models_pytorch as smp
from PIL import Image
from albumentations import Compose, Resize
from albumentations.pytorch import ToTensorV2

# ----------------------------------------------------
# Mask Editor Class
# ----------------------------------------------------
class MaskEditor:
    def __init__(self, image_dir, alternate_image_dir, mask_dir, output_mask_dir, screen_width, screen_height):
        self.image_dir = image_dir
        # Make mask folder optional.
        self.mask_dir = mask_dir.strip() if mask_dir.strip() != "" else None
        self.output_mask_dir = output_mask_dir
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Optional alternate image folder.
        self.second_image_dir = alternate_image_dir.strip() if alternate_image_dir.strip() != "" else None

        os.makedirs(self.output_mask_dir, exist_ok=True)
        self.image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if os.path.isfile(os.path.join(self.image_dir, f))
        ])
        if self.mask_dir:
            self.mask_files = sorted([
                f for f in os.listdir(self.mask_dir)
                if os.path.isfile(os.path.join(self.mask_dir, f))
            ])
            if len(self.image_files) != len(self.mask_files):
                sys.exit("Error: The number of files in the primary image and mask folders are not equal.")
        else:
            self.mask_files = None

        if self.second_image_dir is not None:
            self.second_image_files = sorted([
                f for f in os.listdir(self.second_image_dir)
                if os.path.isfile(os.path.join(self.second_image_dir, f))
            ])
            if len(self.second_image_files) != len(self.image_files):
                sys.exit("Error: The number of files in the alternate image folder does not match the primary image folder.")
        else:
            self.second_image_files = None

        self.current_index = 0

        # Viewing parameters.
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.show_mask_overlay = False

        # Drawing parameters.
        self.is_drawing = False
        self.is_erasing = False
        self.last_point = None
        self.brush_size = 2

        # Line drawing mode.
        self.line_mode_active = False
        self.line_points = []
        self.connected_lines_stack = []

        # Miscellaneous.
        self.show_menu = True
        self.use_alternate = False

        self.cached_image = None         # Primary image (BGR)
        self.cached_image_alt = None     # Alternate image
        self.cached_mask = None          # Final binary mask

        # Attributes for model/drawing management.
        self.drawn_mask = None  
        self.model_mask = None  
        self.combined_mask_float = None

        self.undo_stack = []  

        self.running = True
        self.live_drawing = True

        # --- Model-related attributes ---
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.models = {}   # Loaded models.
        # Hold model checkbox variables (default unchecked).
        self.model_selection_vars = {}
        # Cache predictions to avoid redundancy.
        self.prediction_cache = {}
        self.transform_for_model = Compose([
            Resize(512, 512),
            ToTensorV2()
        ])
        self.inference_threshold = 0.5
        self.load_ensemble_models()

    def load_ensemble_models(self):
        logging.info("Loading ensemble models...")
        self.models.clear()
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
        for model_key, info in hardcoded_models.items():
            folder = info["folder"]
            model = self._load_single_model(folder, info["model_type"], info["encoder"])
            if model is not None:
                self.models[model_key] = model
                logging.info(f" -> Loaded: {model_key}")
        if not self.models:
            logging.error("No models were loaded. Check your weights folder(s).")

    def _load_single_model(self, model_folder, model_type, encoder):
        if not os.path.isdir(model_folder):
            logging.warning(f"Model folder {model_folder} does not exist.")
            return None
        model_files = [f for f in os.listdir(model_folder) if f.endswith("_final.pth")]
        if not model_files:
            logging.warning(f"No '_final.pth' found in {model_folder}.")
            return None
        model_path = os.path.join(model_folder, model_files[0])
        ModelClass = getattr(smp, model_type, None)
        if ModelClass is None:
            logging.error(f"Model type {model_type} not recognized in SMP.")
            return None
        try:
            model = ModelClass(
                encoder_name=encoder,
                encoder_weights=None,
                in_channels=3,
                classes=1,
                activation=None
            ).to(self.device)
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except Exception as e:
            logging.error(f"Error loading model from {model_path}: {e}")
            return None

    def update_binary_mask(self):
        if self.combined_mask_float is not None:
            self.cached_mask = ((self.combined_mask_float > self.inference_threshold) * 255).astype(np.uint8)

    def save_state_to_undo_stack(self):
        if self.cached_mask is not None:
            self.undo_stack.append(self.cached_mask.copy())

    def draw_on_mask(self, event, x, y, flags, param):
        resize_factor_x, resize_factor_y = self.resize_factors
        crop_x1, crop_y1 = self.crop_offsets
        mask_x = int(x / resize_factor_x) + crop_x1
        mask_y = int(y / resize_factor_y) + crop_y1

        if self.line_mode_active:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.line_points.append((mask_x, mask_y))
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.save_state_to_undo_stack()
            self.is_drawing = True
            self.last_point = (mask_x, mask_y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.last_point = None
        elif event == cv2.EVENT_MOUSEMOVE and self.is_drawing:
            if 0 <= mask_x < self.cached_mask.shape[1] and 0 <= mask_y < self.cached_mask.shape[0]:
                value = 0 if self.is_erasing else 255
                if self.last_point is not None:
                    cv2.line(self.cached_mask, self.last_point, (mask_x, mask_y), value, self.brush_size)
                self.last_point = (mask_x, mask_y)

    def connect_points(self):
        if len(self.line_points) > 1:
            self.save_state_to_undo_stack()
            connected_lines = []
            for i in range(1, len(self.line_points)):
                cv2.line(self.cached_mask, self.line_points[i - 1], self.line_points[i], 255, self.brush_size)
                connected_lines.append((self.line_points[i - 1], self.line_points[i]))
            self.connected_lines_stack.append(connected_lines)
        self.line_points = []

    def undo_last_connected_lines(self):
        if self.connected_lines_stack:
            self.save_state_to_undo_stack()
            last_lines = self.connected_lines_stack.pop()
            for line in last_lines:
                cv2.line(self.cached_mask, line[0], line[1], 0, self.brush_size)
            print("Last connected lines undone.")

    def load_image_and_mask(self, image_path, mask_path, alternate_image_path=None):
        self.cached_image = cv2.imread(image_path)
        self.prediction_cache = {}
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        else:
            mask = np.zeros((self.cached_image.shape[0], self.cached_image.shape[1]), dtype=np.uint8)
        self.drawn_mask = mask.copy()
        self.cached_mask = mask.copy()
        self.model_mask = np.zeros_like(mask, dtype=np.uint8)
        self.combined_mask_float = np.maximum(self.drawn_mask, self.model_mask) / 255.0
        self.update_binary_mask()
        self.undo_stack = []
        self.connected_lines_stack = []
        self.line_points = []
        self.live_drawing = True
        self.cached_image_alt = cv2.imread(alternate_image_path) if alternate_image_path is not None else None

    def _run_single_model_inference(self, model):
        image_rgb = cv2.cvtColor(self.cached_image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image_rgb.shape[:2]
        aug = self.transform_for_model(image=image_rgb)
        img_tensor = aug['image'].to(self.device).unsqueeze(0)
        with torch.no_grad():
            pred = model(img_tensor.float())
            pred = torch.sigmoid(pred)
            pred_float = pred.cpu().numpy()[0, 0]
        mask_pil = Image.fromarray((pred_float * 255).astype(np.uint8), mode="L")
        mask_pil = mask_pil.resize((original_w, original_h), resample=Image.NEAREST)
        pred_resized = np.array(mask_pil, dtype=np.uint8) / 255.0
        return pred_resized

    def update_ensemble_mask(self):
        selected_preds = []
        # For each model that is checked, if not cached, run inference.
        for key, model in self.models.items():
            if self.model_selection_vars[key].get():
                if key not in self.prediction_cache:
                    self.prediction_cache[key] = self._run_single_model_inference(model)
                selected_preds.append(self.prediction_cache[key])
        if selected_preds:
            ensemble = np.mean(selected_preds, axis=0)
            self.model_mask = (ensemble * 255).astype(np.uint8)
        else:
            self.model_mask = np.zeros_like(self.cached_mask, dtype=np.uint8)
        self.combined_mask_float = np.maximum(self.drawn_mask, self.model_mask) / 255.0
        self.update_binary_mask()

    def on_model_selection_changed(self, key):
        # Do nothing immediately; ensemble updates occur only when "Run Model" is pressed.
        print(f"Model selection for '{key}' changed. Press 'Run Model' to update ensemble.")

    def run_ensemble_inference_on_current(self):
        if not self.models:
            messagebox.showerror("Error", "No models loaded. Cannot run inference.")
            return
        if self.cached_image is None:
            messagebox.showerror("Error", "No image loaded.")
            return
        self.drawn_mask = self.cached_mask.copy()
        self.update_ensemble_mask()
        logging.info("Model inference applied and merged into current mask.")
        messagebox.showinfo("Done", "Model inference complete. Mask updated.")

    def clear_mask(self):
        if self.cached_mask is not None:
            self.save_state_to_undo_stack()
            self.cached_mask.fill(0)
            logging.info("Mask cleared.")
            messagebox.showinfo("Clear Mask", "Mask cleared successfully.")

    def prev_image(self):
        self.current_index = (self.current_index - 1 + len(self.image_files)) % len(self.image_files)
        self.pan_x, self.pan_y, self.zoom_factor = 0, 0, 1.0
        image_path = os.path.join(self.image_dir, self.image_files[self.current_index])
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, self.mask_files[self.current_index])
        else:
            mask_path = None
        alternate_path = os.path.join(self.second_image_dir, self.second_image_files[self.current_index]) if self.second_image_dir else None
        self.load_image_and_mask(image_path, mask_path, alternate_path)

    def next_image(self):
        self.current_index = (self.current_index + 1) % len(self.image_files)
        self.pan_x, self.pan_y, self.zoom_factor = 0, 0, 1.0
        image_path = os.path.join(self.image_dir, self.image_files[self.current_index])
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, self.mask_files[self.current_index])
        else:
            mask_path = None
        alternate_path = os.path.join(self.second_image_dir, self.second_image_files[self.current_index]) if self.second_image_dir else None
        self.load_image_and_mask(image_path, mask_path, alternate_path)

    def zoom_in(self):
        self.zoom_factor = min(self.zoom_factor * 1.2, 10.0)

    def zoom_out(self):
        self.zoom_factor = max(self.zoom_factor / 1.2, 1.0)

    def pan_up(self):
        self.pan_y = max(self.pan_y - int(400 / self.zoom_factor), -self.cached_image.shape[0] // 2)

    def pan_down(self):
        self.pan_y = min(self.pan_y + int(400 / self.zoom_factor), self.cached_image.shape[0] // 2)

    def pan_left(self):
        self.pan_x = max(self.pan_x - int(400 / self.zoom_factor), -self.cached_image.shape[1] // 2)

    def pan_right(self):
        self.pan_x = min(self.pan_x + int(400 / self.zoom_factor), self.cached_image.shape[1] // 2)

    def toggle_mask_overlay(self):
        self.show_mask_overlay = not self.show_mask_overlay

    def save_mask(self):
        output_path = os.path.join(
            self.output_mask_dir,
            "{}_mask.png".format(os.path.splitext(self.image_files[self.current_index])[0])
        )
        cv2.imwrite(output_path, self.cached_mask)
        print("Modified mask saved to {}".format(output_path))

    def undo(self):
        if self.undo_stack:
            self.cached_mask = self.undo_stack.pop()
            print("Undo performed.")

    def toggle_eraser(self):
        self.is_erasing = not self.is_erasing

    def toggle_line_mode(self):
        self.line_mode_active = not self.line_mode_active
        self.line_points = []
        print("Line mode toggled: now", "ON" if self.line_mode_active else "OFF")

    def connect_points_cmd(self):
        if self.line_mode_active:
            self.connect_points()

    def toggle_menu(self):
        self.show_menu = not self.show_menu

    def switch_view(self):
        if self.second_image_dir is not None:
            self.use_alternate = not self.use_alternate

    def quit_editor(self):
        self.quit_app()

    def quit_app(self):
        print("Exiting application...")
        self.running = False
        cv2.destroyAllWindows()

    def display_image(self):
        # Remove the overlay of key instructions.
        if self.use_alternate and self.cached_image_alt is not None:
            base_image = self.cached_image_alt.copy()
        else:
            base_image = self.cached_image.copy()

        if self.show_mask_overlay:
            overlay = base_image.copy()
            overlay[self.cached_mask > 0] = [0, 255, 0]
            combined = overlay
        else:
            combined = base_image

        height, width = combined.shape[:2]
        crop_width = int(width / self.zoom_factor)
        crop_height = int(height / self.zoom_factor)
        center_x, center_y = width // 2, height // 2

        x1 = max(0, min(center_x - crop_width // 2 + self.pan_x, width - crop_width))
        y1 = max(0, min(center_y - crop_height // 2 + self.pan_y, height - crop_height))
        x2, y2 = x1 + crop_width, y1 + crop_height

        cropped = combined[y1:y2, x1:x2]
        cropped_mask = self.cached_mask[y1:y2, x1:x2]

        scale_factor = min(self.screen_width / cropped.shape[1], self.screen_height / cropped.shape[0])
        new_width = int(cropped.shape[1] * scale_factor)
        new_height = int(cropped.shape[0] * scale_factor)
        resized = cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(cropped_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        self.resize_factors = (new_width / cropped.shape[1], new_height / cropped.shape[0])
        self.crop_offsets = (x1, y1)

        if self.show_mask_overlay:
            resized[resized_mask > 0] = [0, 255, 0]

        if self.line_mode_active:
            for point in self.line_points:
                display_x = int((point[0] - x1) * self.resize_factors[0])
                display_y = int((point[1] - y1) * self.resize_factors[1])
                cv2.circle(resized, (display_x, display_y), 5, (0, 0, 255), -1)

        cv2.imshow('Image Viewer', resized)

    def run(self):
        image_path = os.path.join(self.image_dir, self.image_files[self.current_index])
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, self.mask_files[self.current_index])
        else:
            mask_path = None
        alternate_path = os.path.join(self.second_image_dir, self.second_image_files[self.current_index]) if self.second_image_dir else None
        self.load_image_and_mask(image_path, mask_path, alternate_path)
        cv2.namedWindow('Image Viewer')
        cv2.setMouseCallback('Image Viewer', self.draw_on_mask)

        self.running = True
        while self.running:
            self.display_image()
            key = cv2.waitKey(1) & 0xFF

            if key == ord('a'):
                self.prev_image()
            elif key == ord('d'):
                self.next_image()
            elif key == ord('p'):
                self.connect_points_cmd()
            elif key == ord('+'):
                self.zoom_in()
            elif key == ord('-'):
                self.zoom_out()
            elif key == ord('i'):
                self.pan_up()
            elif key == ord('k'):
                self.pan_down()
            elif key == ord('j'):
                self.pan_left()
            elif key == ord('l'):
                self.pan_right()
            elif key == ord('v'):
                self.switch_view()
            elif key == ord('t'):
                self.toggle_mask_overlay()
            elif key == ord('s'):
                self.save_mask()
            elif key == ord('z'):
                if self.line_mode_active and self.connected_lines_stack:
                    self.undo_last_connected_lines()
                elif self.undo_stack:
                    self.cached_mask = self.undo_stack.pop()
                    print("Undo performed.")
            elif key == ord('e'):
                self.toggle_eraser()
            elif key == ord('o'):
                self.toggle_line_mode()
            elif key == ord('f'):
                self.brush_size = min(self.brush_size + 1, 50)
                print("Brush size increased to {}".format(self.brush_size))
            elif key == ord('g'):
                self.brush_size = max(self.brush_size - 1, 1)
                print("Brush size decreased to {}".format(self.brush_size))
            elif key == 9:  # TAB key
                self.toggle_menu()
            elif key == 27:  # ESC key
                self.quit_editor()

# ----------------------------------------------------
# Control Panel (Tkinter)
# ----------------------------------------------------
def start_control_panel(editor):
    def panel():
        cp = tk.Tk()
        cp.title("Control Panel")

        # Section 1: Navigation
        nav_frame = tk.LabelFrame(cp, text="Navigation", padx=5, pady=5)
        nav_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        tk.Button(nav_frame, text="Prev (a)", command=editor.prev_image, width=20).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(nav_frame, text="Next (d)", command=editor.next_image, width=20).grid(row=0, column=1, padx=5, pady=5)

        # Section 2: Zoom
        zoom_frame = tk.LabelFrame(cp, text="Zoom", padx=5, pady=5)
        zoom_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        tk.Button(zoom_frame, text="Zoom In (+)", command=editor.zoom_in, width=10).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(zoom_frame, text="Zoom Out (-)", command=editor.zoom_out, width=10).grid(row=0, column=1, padx=5, pady=5)

        # Section 3: Pan (Compass)
        pan_frame = tk.LabelFrame(cp, text="Pan", padx=5, pady=5)
        pan_frame.grid(row=2, column=0, padx=5, pady=5)
        tk.Button(pan_frame, text="Up (i)", command=editor.pan_up, width=5).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(pan_frame, text="Left (j)", command=editor.pan_left, width=5).grid(row=1, column=0, padx=5, pady=5)
        tk.Button(pan_frame, text="Right (l)", command=editor.pan_right, width=5).grid(row=1, column=2, padx=5, pady=5)
        tk.Button(pan_frame, text="Down (k)", command=editor.pan_down, width=5).grid(row=2, column=1, padx=5, pady=5)

        # Section 4: Brush Size Slider
        brush_frame = tk.LabelFrame(cp, text="Brush Size", padx=5, pady=5)
        brush_frame.grid(row=3, column=0, padx=5, pady=5, sticky="ew")
        brush_slider = tk.Scale(brush_frame, from_=1, to=50, orient=tk.HORIZONTAL, length=200)
        brush_slider.set(editor.brush_size)
        brush_slider.grid(row=0, column=0, padx=5, pady=5)
        def update_brush_size(val):
            editor.brush_size = int(val)
        brush_slider.config(command=update_brush_size)

        # Section 5: Inference Threshold Slider
        thresh_frame = tk.LabelFrame(cp, text="Inference Threshold", padx=5, pady=5)
        thresh_frame.grid(row=4, column=0, padx=5, pady=5, sticky="ew")
        thresh_slider = tk.Scale(thresh_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, length=200)
        thresh_slider.set(editor.inference_threshold)
        thresh_slider.grid(row=0, column=0, padx=5, pady=5)
        def update_threshold(val):
            editor.inference_threshold = float(val)
            editor.update_binary_mask()
        thresh_slider.config(command=update_threshold)

        # Section 6: Model Selection
        model_frame = tk.LabelFrame(cp, text="Model Selection", padx=5, pady=5)
        model_frame.grid(row=5, column=0, padx=5, pady=5, sticky="ew")
        for model_key in editor.models.keys():
            if model_key not in editor.model_selection_vars:
                editor.model_selection_vars[model_key] = tk.BooleanVar(value=False)
            tk.Checkbutton(model_frame, text=model_key, variable=editor.model_selection_vars[model_key]).pack(anchor="w")

        # Section 7: Other Commands
        other_frame = tk.LabelFrame(cp, text="Other Commands", padx=5, pady=5)
        other_frame.grid(row=6, column=0, padx=5, pady=5, sticky="ew")
        tk.Button(other_frame, text="Toggle Mask (t)", command=editor.toggle_mask_overlay, width=20).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(other_frame, text="Save Mask (s)", command=editor.save_mask, width=20).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(other_frame, text="Undo (z)", command=editor.undo, width=20).grid(row=1, column=0, padx=5, pady=5)
        tk.Button(other_frame, text="Toggle Eraser (e)", command=editor.toggle_eraser, width=20).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(other_frame, text="Toggle Line Mode (o)", command=editor.toggle_line_mode, width=20).grid(row=2, column=0, padx=5, pady=5)
        tk.Button(other_frame, text="Connect Points (p)", command=editor.connect_points_cmd, width=20).grid(row=2, column=1, padx=5, pady=5)
        tk.Button(other_frame, text="Run Model (r)", command=editor.run_ensemble_inference_on_current, width=20).grid(row=3, column=0, padx=5, pady=5)
        tk.Button(other_frame, text="Switch View (v)", command=editor.switch_view, width=20).grid(row=3, column=1, padx=5, pady=5)
        tk.Button(other_frame, text="Clear Mask (c)", command=editor.clear_mask, width=20).grid(row=4, column=0, padx=5, pady=5)
        tk.Button(other_frame, text="Quit (ESC)", command=editor.quit_editor, width=20).grid(row=4, column=1, padx=5, pady=5)

        cp.mainloop()
    threading.Thread(target=panel, daemon=True).start()

# ----------------------------------------------------
# Configuration GUI
# ----------------------------------------------------
def open_config_window():
    config = {}
    root = tk.Tk()
    root.title("Mask Editor Configuration")

    image_dir_var = tk.StringVar()
    alternate_dir_var = tk.StringVar()
    # Mask folder is optional.
    mask_dir_var = tk.StringVar()
    output_dir_var = tk.StringVar()
    screen_width_var = tk.IntVar(value=1920)
    screen_height_var = tk.IntVar(value=900)

    def browse_image_dir():
        selected = filedialog.askdirectory(title="Select Primary Image Folder")
        if selected:
            image_dir_var.set(selected)

    def browse_alternate_dir():
        selected = filedialog.askdirectory(title="Select Alternate Image Folder (Optional)")
        if selected:
            alternate_dir_var.set(selected)

    def browse_mask_dir():
        selected = filedialog.askdirectory(title="Select Mask Folder (Optional)")
        if selected:
            mask_dir_var.set(selected)

    def browse_output_dir():
        selected = filedialog.askdirectory(title="Select Output Folder")
        if selected:
            output_dir_var.set(selected)

    def submit():
        if not image_dir_var.get() or not output_dir_var.get():
            tk.messagebox.showerror("Error", "Please select primary image and output folders.")
            return
        config['image_dir'] = image_dir_var.get()
        config['mask_dir'] = mask_dir_var.get()
        config['output_mask_dir'] = output_dir_var.get()
        config['screen_width'] = screen_width_var.get()
        config['screen_height'] = screen_height_var.get()
        config['alternate_dir'] = alternate_dir_var.get()
        root.destroy()

    tk.Label(root, text="Primary Image Folder:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    tk.Entry(root, textvariable=image_dir_var, width=50).grid(row=0, column=1, padx=5, pady=5)
    tk.Button(root, text="Browse", command=browse_image_dir).grid(row=0, column=2, padx=5, pady=5)
    tk.Label(root, text="Alternate Image Folder (Optional):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
    tk.Entry(root, textvariable=alternate_dir_var, width=50).grid(row=1, column=1, padx=5, pady=5)
    tk.Button(root, text="Browse", command=browse_alternate_dir).grid(row=1, column=2, padx=5, pady=5)
    tk.Label(root, text="Mask Folder (Optional):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
    tk.Entry(root, textvariable=mask_dir_var, width=50).grid(row=2, column=1, padx=5, pady=5)
    tk.Button(root, text="Browse", command=browse_mask_dir).grid(row=2, column=2, padx=5, pady=5)
    tk.Label(root, text="Output Folder:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
    tk.Entry(root, textvariable=output_dir_var, width=50).grid(row=3, column=1, padx=5, pady=5)
    tk.Button(root, text="Browse", command=browse_output_dir).grid(row=3, column=2, padx=5, pady=5)
    tk.Label(root, text="Screen Width:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
    tk.Scale(root, from_=800, to=3840, orient=tk.HORIZONTAL, variable=screen_width_var).grid(row=4, column=1, padx=5, pady=5, sticky="ew")
    tk.Label(root, text="Screen Height:").grid(row=5, column=0, sticky="w", padx=5, pady=5)
    tk.Scale(root, from_=600, to=2160, orient=tk.HORIZONTAL, variable=screen_height_var).grid(row=5, column=1, padx=5, pady=5, sticky="ew")
    tk.Button(root, text="Launch Editor", command=submit).grid(row=6, column=1, padx=5, pady=15)

    root.mainloop()
    return config

# ----------------------------------------------------
# Main Entry Point
# ----------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    config = open_config_window()
    editor = MaskEditor(
        config['image_dir'],
        config['alternate_dir'],
        config['mask_dir'],
        config['output_mask_dir'],
        config['screen_width'],
        config['screen_height']
    )
    start_control_panel(editor)
    editor.run()

if __name__ == "__main__":
    main()
