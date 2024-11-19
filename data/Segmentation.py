import os
import shutil
from histolab.slide import Slide
from histolab.tiler import GridTiler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def configure_tiler(tile_size=(224, 224), level=0, tissue_percent=60, pixel_overlap=0, suffix=".png"):
    """
    Configures and returns a GridTiler object.
    """
    return GridTiler(
        tile_size=tile_size,
        level=level,
        check_tissue=True,
        tissue_percent=tissue_percent,
        pixel_overlap=pixel_overlap,
        prefix="",
        suffix=suffix
    )

def prepare_output_directory(base_path, slide_name):
    """
    Ensures the output directory for tiles is ready.
    If the directory exists, it is deleted and recreated.
    """
    output_path = os.path.join(base_path, slide_name)
    if os.path.exists(output_path):
        logging.warning(f"Output directory exists. Removing: {output_path}")
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    logging.info(f"Created directory: {output_path}")
    return output_path

def process_slide(grid_tiler, slide_path, output_path):
    """
    Processes a single Whole Slide Image (WSI) using GridTiler.
    """
    try:
        slide = Slide(slide_path, output_path)
        grid_tiler.extract(slide)
        logging.info(f"Finished tiling for: {slide_path}")
    except Exception as e:
        logging.error(f"Failed to process slide {slide_path}: {e}")

def tiling_wsi(input_path="output/", output_base_path="Tile_Save/"):
    """
    Iterates over WSIs in the input directory, processes each with tiling,
    and saves the tiles in the specified output directory.
    """
    # Configure the tiler
    grid_tiler = configure_tiler()

    # Ensure the output base directory exists
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
        logging.info(f"Created base output directory: {output_base_path}")

    # Process each slide in the input directory
    slide_count = 0
    for slide_file in os.listdir(input_path):
        slide_path = os.path.join(input_path, slide_file)
        slide_name = os.path.splitext(slide_file)[0]
        logging.info(f"Processing slide: {slide_file}")

        # Prepare output directory for the current slide
        output_path = prepare_output_directory(output_base_path, slide_name)

        # Process the slide
        process_slide(grid_tiler, slide_path, output_path)

        slide_count += 1
        logging.info(f"Processed {slide_count} slide(s) so far.")

    logging.info(f"Completed processing all slides. Total slides processed: {slide_count}")

if __name__ == "__main__":
    tiling_wsi()
