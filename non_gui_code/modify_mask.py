#!/usr/bin/env python3
import cv2
import os
import sys
import numpy as np
import argparse

class MaskEditor:
    def __init__(self, image_dir, mask_dir, output_mask_dir, screen_width, screen_height):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.output_mask_dir = output_mask_dir
        self.screen_width = screen_width
        self.screen_height = screen_height

        os.makedirs(self.output_mask_dir, exist_ok=True)
        self.image_files = sorted(
            [f for f in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f))]
        )
        self.mask_files = sorted(
            [f for f in os.listdir(self.mask_dir) if os.path.isfile(os.path.join(self.mask_dir, f))]
        )

        # Check if both folders have the same files
        if self.image_files != self.mask_files:
            sys.exit("Error: The files in the image and mask folders do not match exactly.")

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
        self.cached_image = None
        self.cached_mask = None
        self.original_mask = None
        self.resize_factors = (1.0, 1.0)  # (width_factor, height_factor)
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
                # In line mode, add points rather than immediate drawing.
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

    def load_image_and_mask(self, image_path, mask_path):
        self.cached_image = cv2.imread(image_path)
        self.cached_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # Ensure the mask is binary
        self.cached_mask = np.where(self.cached_mask > 0, 255, 0).astype(np.uint8)
        self.original_mask = self.cached_mask.copy()
        self.undo_stack = []
        self.connected_lines_stack = []

    def display_image(self):
        # Prepare the overlay
        if self.show_mask_overlay:
            overlay = self.cached_image.copy()
            overlay[self.cached_mask > 0] = [0, 255, 0]  # Green overlay for mask
            combined = overlay
        else:
            combined = self.cached_image

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
                "Keys: A-Previous | D-Next | +/- Zoom | IJKL-Pan | T-Toggle Mask | S-Save | "
                "Z-Undo | E-Toggle Eraser | O-Toggle Line Mode | F-Increase Brush | G-Decrease Brush (Current: {}) | "
                "P-Connect Points | TAB-Show/Hide Menu | ESC-Quit".format(self.brush_size)
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
        """Function to quit the application gracefully."""
        print("Exiting application...")
        self.running = False
        cv2.destroyAllWindows()

    def run(self):
        # Load the first image and mask
        image_path = os.path.join(self.image_dir, self.image_files[self.current_index])
        mask_path = os.path.join(self.mask_dir, self.mask_files[self.current_index])
        self.load_image_and_mask(image_path, mask_path)
        cv2.namedWindow('Image Viewer')
        cv2.setMouseCallback('Image Viewer', self.draw_on_mask)

        self.running = True
        while self.running:
            self.display_image()
            key = cv2.waitKey(1) & 0xFF

            # Navigation: A for previous image, D for next image
            if key == ord('a'):
                self.current_index = (self.current_index - 1 + len(self.image_files)) % len(self.image_files)
                self.pan_x, self.pan_y, self.zoom_factor = 0, 0, 1.0
                image_path = os.path.join(self.image_dir, self.image_files[self.current_index])
                mask_path = os.path.join(self.mask_dir, self.mask_files[self.current_index])
                self.load_image_and_mask(image_path, mask_path)

            elif key == ord('d'):
                self.current_index = (self.current_index + 1) % len(self.image_files)
                self.pan_x, self.pan_y, self.zoom_factor = 0, 0, 1.0
                image_path = os.path.join(self.image_dir, self.image_files[self.current_index])
                mask_path = os.path.join(self.mask_dir, self.mask_files[self.current_index])
                self.load_image_and_mask(image_path, mask_path)

            elif key == ord('p'):  # Connect points in line mode
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
            elif key == 9:  # TAB key toggles menu visibility
                self.show_menu = not self.show_menu
            elif key == 27:  # ESC key to quit
                self.quit_app()

def main():
    parser = argparse.ArgumentParser(description="Interactive Mask Editor")
    parser.add_argument(
        "--image_dir", type=str, default="./data/val/images/",
        help="Path to the directory containing images."
    )
    parser.add_argument(
        "--mask_dir", type=str, default="./data/val/masks/",
        help="Path to the directory containing masks."
    )
    parser.add_argument(
        "--output_mask_dir", type=str, default="./val/modified_masks/",
        help="Directory where modified masks will be saved."
    )
    parser.add_argument(
        "--screen_width", type=int, default=1920,
        help="Screen width for display."
    )
    parser.add_argument(
        "--screen_height", type=int, default=900,
        help="Screen height for display."
    )
    args = parser.parse_args()

    editor = MaskEditor(
        args.image_dir, args.mask_dir, args.output_mask_dir,
        args.screen_width, args.screen_height
    )
    editor.run()

if __name__ == "__main__":
    main()
