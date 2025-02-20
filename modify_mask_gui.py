#!/usr/bin/env python3
import cv2
import os
import sys
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

# -----------------------
# Mask Editor Class
# -----------------------
class MaskEditor:
    def __init__(self, image_dir, alternate_image_dir, mask_dir, output_mask_dir, screen_width, screen_height):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.output_mask_dir = output_mask_dir
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Optional alternate image folder
        self.second_image_dir = alternate_image_dir.strip() if alternate_image_dir.strip() != "" else None

        os.makedirs(self.output_mask_dir, exist_ok=True)
        self.image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if os.path.isfile(os.path.join(self.image_dir, f))
        ])
        self.mask_files = sorted([
            f for f in os.listdir(self.mask_dir)
            if os.path.isfile(os.path.join(self.mask_dir, f))
        ])

        # Check that the number of files in primary image and mask folders are equal.
        if len(self.image_files) != len(self.mask_files):
            sys.exit("Error: The number of files in the primary image and mask folders are not equal.")

        # If alternate folder is provided, list and check that its file count matches.
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

        # Viewing parameters
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.show_mask_overlay = False

        # Drawing parameters
        self.is_drawing = False
        self.is_erasing = False  # Eraser mode
        self.last_point = None
        self.brush_size = 2

        # Line drawing mode
        self.line_mode_active = False
        self.line_points = []
        self.connected_lines_stack = []

        # Miscellaneous
        self.show_menu = True
        self.use_alternate = False  # Flag for toggling view

        self.cached_image = None         # Primary image
        self.cached_image_alt = None     # Alternate image
        self.cached_mask = None
        self.original_mask = None
        self.resize_factors = (1.0, 1.0)   # (width_factor, height_factor)
        self.crop_offsets = (0, 0)
        self.undo_stack = []

        # Running flag for main loop
        self.running = True

    def save_state_to_undo_stack(self):
        if self.cached_mask is not None:
            self.undo_stack.append(self.cached_mask.copy())

    def draw_on_mask(self, event, x, y, flags, param):
        resize_factor_x, resize_factor_y = self.resize_factors
        crop_x1, crop_y1 = self.crop_offsets
        mask_x = int(x / resize_factor_x) + crop_x1
        mask_y = int(y / resize_factor_y) + crop_y1

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.line_mode_active:
                self.line_points.append((mask_x, mask_y))
            else:
                self.save_state_to_undo_stack()
                self.is_drawing = True
                self.last_point = (mask_x, mask_y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.last_point = None
        elif event == cv2.EVENT_MOUSEMOVE and self.is_drawing:
            if 0 <= mask_x < self.cached_mask.shape[1] and 0 <= mask_y < self.cached_mask.shape[0]:
                color = 0 if self.is_erasing else 255
                if self.last_point is not None:
                    cv2.line(self.cached_mask, self.last_point, (mask_x, mask_y), color, self.brush_size)
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
        self.cached_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        self.cached_mask = np.where(self.cached_mask > 0, 255, 0).astype(np.uint8)
        self.original_mask = self.cached_mask.copy()
        self.undo_stack = []
        self.connected_lines_stack = []
        if alternate_image_path is not None:
            self.cached_image_alt = cv2.imread(alternate_image_path)
        else:
            self.cached_image_alt = None

    def display_image(self):
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

        if self.show_menu:
            eraser_status = "Eraser ON" if self.is_erasing else "Eraser OFF"
            line_mode_status = "Line Mode ON" if self.line_mode_active else "Line Mode OFF"
            keys_info = (
                "Keys: A-Prev | D-Next | +/- Zoom | IJKL-Pan | T-Toggle Mask | S-Save | "
                "Z-Undo | E-Toggle Eraser | O-Toggle Line Mode | F-Increase Brush | G-Decrease Brush ({} px) | "
                "P-Connect Points | TAB-Show/Hide Menu | V-Switch View | ESC-Quit".format(self.brush_size)
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (255, 255, 255)
            thickness = 1
            y0, dy = 20, 20
            for i, line in enumerate(keys_info.split(" | ")):
                y = y0 + i * dy
                cv2.putText(resized, line, (10, y), font, font_scale, font_color, thickness, cv2.LINE_AA)

        cv2.imshow('Image Viewer', resized)

    def quit_app(self):
        print("Exiting application...")
        self.running = False
        cv2.destroyAllWindows()

    def run(self):
        image_path = os.path.join(self.image_dir, self.image_files[self.current_index])
        mask_path = os.path.join(self.mask_dir, self.mask_files[self.current_index])
        alternate_path = None
        if self.second_image_dir is not None:
            alternate_path = os.path.join(self.second_image_dir, self.second_image_files[self.current_index])
        self.load_image_and_mask(image_path, mask_path, alternate_path)
        cv2.namedWindow('Image Viewer')
        cv2.setMouseCallback('Image Viewer', self.draw_on_mask)

        self.running = True
        while self.running:
            self.display_image()
            key = cv2.waitKey(1) & 0xFF

            if key == ord('a'):
                self.current_index = (self.current_index - 1 + len(self.image_files)) % len(self.image_files)
                self.pan_x, self.pan_y, self.zoom_factor = 0, 0, 1.0
                image_path = os.path.join(self.image_dir, self.image_files[self.current_index])
                mask_path = os.path.join(self.mask_dir, self.mask_files[self.current_index])
                alternate_path = None
                if self.second_image_dir is not None:
                    alternate_path = os.path.join(self.second_image_dir, self.second_image_files[self.current_index])
                self.load_image_and_mask(image_path, mask_path, alternate_path)

            elif key == ord('d'):
                self.current_index = (self.current_index + 1) % len(self.image_files)
                self.pan_x, self.pan_y, self.zoom_factor = 0, 0, 1.0
                image_path = os.path.join(self.image_dir, self.image_files[self.current_index])
                mask_path = os.path.join(self.mask_dir, self.mask_files[self.current_index])
                alternate_path = None
                if self.second_image_dir is not None:
                    alternate_path = os.path.join(self.second_image_dir, self.second_image_files[self.current_index])
                self.load_image_and_mask(image_path, mask_path, alternate_path)

            elif key == ord('p'):
                if self.line_mode_active:
                    self.connect_points()

            elif key == ord('+'):
                self.zoom_factor = min(self.zoom_factor * 1.2, 10.0)
            elif key == ord('-'):
                self.zoom_factor = max(self.zoom_factor / 1.2, 1.0)
            elif key == ord('i'):
                self.pan_y = max(self.pan_y - int(400 / self.zoom_factor), -self.cached_image.shape[0] // 2)
            elif key == ord('k'):
                self.pan_y = min(self.pan_y + int(400 / self.zoom_factor), self.cached_image.shape[0] // 2)
            elif key == ord('j'):
                self.pan_x = max(self.pan_x - int(400 / self.zoom_factor), -self.cached_image.shape[1] // 2)
            elif key == ord('l'):
                self.pan_x = min(self.pan_x + int(400 / self.zoom_factor), self.cached_image.shape[1] // 2)
            elif key == ord('v'):
                if self.second_image_dir is not None:
                    self.use_alternate = not self.use_alternate
            elif key == ord('t'):
                self.show_mask_overlay = not self.show_mask_overlay
            elif key == ord('s'):
                output_path = os.path.join(
                    self.output_mask_dir,
                    "{}_mask.png".format(os.path.splitext(self.image_files[self.current_index])[0])
                )
                cv2.imwrite(output_path, self.cached_mask)
                print("Modified mask saved to {}".format(output_path))
            elif key == ord('z'):
                if self.line_mode_active and self.connected_lines_stack:
                    self.undo_last_connected_lines()
                elif self.undo_stack:
                    self.cached_mask = self.undo_stack.pop()
                    print("Undo performed.")
            elif key == ord('e'):
                self.is_erasing = not self.is_erasing
            elif key == ord('o'):
                self.line_mode_active = not self.line_mode_active
            elif key == ord('f'):
                self.brush_size = min(self.brush_size + 1, 50)
                print("Brush size increased to {}".format(self.brush_size))
            elif key == ord('g'):
                self.brush_size = max(self.brush_size - 1, 1)
                print("Brush size decreased to {}".format(self.brush_size))
            elif key == 9:
                self.show_menu = not self.show_menu
            elif key == 27:
                self.quit_app()

# -----------------------
# Configuration GUI
# -----------------------
def open_config_window():
    config = {}
    root = tk.Tk()
    root.title("Mask Editor Configuration")

    image_dir_var = tk.StringVar()
    alternate_dir_var = tk.StringVar()
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
        selected = filedialog.askdirectory(title="Select Mask Folder")
        if selected:
            mask_dir_var.set(selected)

    def browse_output_dir():
        selected = filedialog.askdirectory(title="Select Output Folder")
        if selected:
            output_dir_var.set(selected)

    def submit():
        if not image_dir_var.get() or not mask_dir_var.get() or not output_dir_var.get():
            tk.messagebox.showerror("Error", "Please select primary image, mask, and output folders.")
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

    tk.Label(root, text="Mask Folder:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
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

# -----------------------
# Main Entry Point
# -----------------------
def main():
    config = open_config_window()
    editor = MaskEditor(
        config['image_dir'],
        config['alternate_dir'],
        config['mask_dir'],
        config['output_mask_dir'],
        config['screen_width'],
        config['screen_height']
    )
    editor.run()

if __name__ == "__main__":
    main()
