import arcpy
from arcpy.ia import ExportTrainingDataForDeepLearning
import math

# Check out the ArcGIS Image Analyst extension
arcpy.CheckOutExtension("ImageAnalyst")
arcpy.env.cellSize = 1  # Set cell size in map units

# Define input parameters
input_raster = "World Imagery"
additional_input_raster = "LidarProducts/DSM_HS"
output_folder = r"E:\Desktop\roads_m2\public_repo\LoggingRoads\dsm_val"
image_chip_format = "TIFF"
metadata_format = "Classified_Tiles"
reference_system = "MAP_SPACE"
output_nofeature_tiles = "ALL_TILES"

# Tile and stride sizes (in map units)
# With cellSize=1, a tile size of 4000 map units produces a 4000×4000-pixel image.
tile_size_x = 4000
tile_size_y = 4000
stride_x = 4000
stride_y = 4000

# The allowed maximum tile size (used for extent adjustment) in map units
tile_width = 4000
tile_height = 4000

# Retrieve the extent of the 'test' layer
layer_name = "test"
layer_extent = arcpy.Describe(layer_name).extent

# Extract and round the original valid extent coordinates
orig_left = round(layer_extent.XMin, 2)
orig_bottom = round(layer_extent.YMin, 2)
orig_right = round(layer_extent.XMax, 2)
orig_top = round(layer_extent.YMax, 2)
print(f"Original valid extent: {orig_left}, {orig_bottom}, {orig_right}, {orig_top}")

# Use the original valid extent as a starting point
extent_left = orig_left
extent_bottom = orig_bottom
extent_right = orig_right
extent_top = orig_top

# --- Extend the overall extent to a multiple of tile_width/tile_height, if desired ---
extendExtent = False  # Set to True to extend the extent

if extendExtent:
    original_width = extent_right - extent_left
    original_height = extent_top - extent_bottom
    # Calculate the required multiples
    new_width = math.ceil(original_width / tile_width) * tile_width
    new_height = math.ceil(original_height / tile_height) * tile_height
    # Determine the extra width/height needed
    extra_width = new_width - original_width
    extra_height = new_height - original_height
    # Adjust the extent symmetrically so the valid area is centered
    extent_left = extent_left - extra_width / 2.0
    extent_bottom = extent_bottom - extra_height / 2.0
    extent_right = extent_right + extra_width / 2.0
    extent_top = extent_top + extra_height / 2.0
    print(f"Extended extent: {extent_left}, {extent_bottom}, {extent_right}, {extent_top}")

# Calculate total number of tiles based on the (extended) extent and tile dimensions (in map units)
num_tiles_x = int((extent_right - extent_left) / tile_width)
num_tiles_y = int((extent_top - extent_bottom) / tile_height)
total_tiles = num_tiles_x * num_tiles_y
print(f"Total number of tiles to process: {total_tiles}")

tiles_processed = 0

# Iterate over the extended extent in steps of tile_width and tile_height
current_x = extent_left
while current_x < extent_right:
    current_y = extent_bottom
    while current_y < extent_top:
        # Define the current tile extent (in map units)
        tile_min_x = round(current_x, 2)
        tile_min_y = round(current_y, 2)
        tile_max_x = round(min(current_x + tile_width, extent_right), 2)
        tile_max_y = round(min(current_y + tile_height, extent_top), 2)

        extent_string = f"{tile_min_x} {tile_min_y} {tile_max_x} {tile_max_y}"
        arcpy.env.extent = extent_string
        print(f"\nProcessing tile: {extent_string}")

        try:
            ExportTrainingDataForDeepLearning(
                in_raster=input_raster,
                in_raster2=additional_input_raster,
                out_folder=output_folder,
                image_chip_format=image_chip_format,
                tile_size_x=tile_size_x,
                tile_size_y=tile_size_y,
                stride_x=stride_x,
                stride_y=stride_y,
                buffer_radius=1,
                output_nofeature_tiles=output_nofeature_tiles,
                metadata_format=metadata_format,
                reference_system=reference_system
            )
            print("Tile processed successfully.")
        except Exception as e:
            print(f"Error processing tile {extent_string}: {e}")

        tiles_processed += 1
        print(f"Progress: {tiles_processed}/{total_tiles} tiles processed.")

        current_y += tile_height  # Move to next tile in Y direction
    current_x += tile_width       # Move to next row of tiles in X direction

print("\nExport completed successfully.")
