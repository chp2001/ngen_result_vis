import os, sys
from typing import Callable, Tuple, Dict, List, TypeAlias, Optional
from warnings import warn
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.cm as cm

import math
import PIL
from PIL import Image, ImageDraw, ImageFont

import time

# Attempting to convert the image code from the original `analyze_output.py`
# into a more self-contained and coherent form.

# The original worked, but makes me feel like I was insane when I wrote it.


# def find_reasonable_aspect_ratio(
#     group_size, group_count, width_grid=2, height_grid=1, resolution_mult=2
# ):
#     """
#     Using the provided information, find a reasonable aspect ratio
#     Mostly by taking the square root and then choosing the nearest common aspect ratio
#     """
#     total_cells = group_size * group_count
#     total_pixels = total_cells * width_grid * height_grid * resolution_mult**2
#     # find the square root
#     cell_sqrt = math.sqrt(total_pixels)

#     common_aspect_ratios = [
#         (854, 480),  # 16:9
#         (1024, 576),
#         (1280, 720),
#         (1366, 768),
#         (1600, 900),
#         (1920, 1080),
#         (2560, 1440),
#         (3840, 2160),
#         (7680, 4320),
#         (15360, 8640),
#     ]
#     aspect_sqrts = [math.sqrt(x[0] * x[1]) for x in common_aspect_ratios]
#     grouped_aspect = [(x, y) for x, y in zip(common_aspect_ratios, aspect_sqrts)]
#     nearest_func = lambda x: abs(x[1] - cell_sqrt)
#     grouped_aspect.sort(key=nearest_func)
#     print(f"cell_sqrt: {cell_sqrt}")
#     for i, (ratio, aspect) in enumerate(grouped_aspect):
#         print(f"aspect: {aspect}, ratio: {ratio}")
#     return grouped_aspect[0][0]

# This function is genuinely confusing.
# The common aspect ratios are interesting and useful, but
# the relation to the parameters is very unclear.
# Group size is obviously the column height,
# group count is the number of columns,
# but what do the remaining parts mean?
# Width grid might be coherent as the number of cells per row *within* an item of
# the column?
# Height then would be the number of rows within an item of the column?
# Together that explains why they are multiplied to get the pixels per item.

# Then resolution mult is the width/height of each cell in pixels?
# That would explain the squared term.

# Overall, this seems to be a very crude estimation of the total pixel count,
# especially when it may significantly over or under estimate the actual
# size produced by the image generation code...

# Is it even useful?
# It might be fine for now to provide only the aspect ratio matching

# Seems unlikely to be useful in practice.

common_16_9_aspect_ratios: List[Tuple[int, int]] = [
    (854, 480),  # 16:9
    (1024, 576),
    (1280, 720),
    (1366, 768),
    (1600, 900),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
    (7680, 4320),
    (15360, 8640),
]


def find_reasonable_aspect_ratio(
    expected_pixels: int,
    aspect_ratios: List[Tuple[int, int]] = common_16_9_aspect_ratios,
) -> Tuple[int, int]:
    """
    Using the provided information, find a reasonable aspect ratio
    by choosing the nearest common aspect ratio to the expected pixel count.
    """
    for i, ratio in enumerate(aspect_ratios):
        if math.prod(ratio) > expected_pixels:
            return aspect_ratios[max(0, i - 1)]
    warn("Could not find a reasonable aspect ratio; returning the largest available.")
    return aspect_ratios[-1]

# All of the functions seem to be trying to do significantly more than
# they need to for their own purpose.

# get_allocated_space does some interesting planning,
# but is doing far too much of the actual image calculations

# Let's try breaking it down into smaller pieces.

def calc_slot_size(
    cell_size: int = 4, width_grid: int = 2, height_grid: int = 1, horiz_divider_width: int = 1, vert_divider_width: int = 1, **kwargs
)->Tuple[int, int]:
    """
    Calculate the pixel dimensions of a single slot based on grid and divider sizes.
    
    Does not account for external padding or margins.
    """
    slot_width = (cell_size * width_grid + (width_grid - 1) * vert_divider_width)
    slot_height = (cell_size * height_grid + (height_grid - 1) * horiz_divider_width)
    return slot_width, slot_height

def calc_column_size(
    group_size: int,
    slot_size: Tuple[int, int],
    horiz_divider_width: int = 1, **kwargs
) -> Tuple[int, int]:
    """
    Calculate the pixel dimensions of a single column based on group size and slot size.
    
    Does not account for external padding or margins.
    """
    slot_width, slot_height = slot_size
    column_width = slot_width
    column_height = group_size * slot_height + (group_size - 1) * horiz_divider_width
    return column_width, column_height

def calc_total_image_size(
    group_count: int,
    column_size: Tuple[int, int],
    vert_divider_width: int = 1,
    vert_padding: int = 6, # difficult to tell where the value came from
    **kwargs
) -> Tuple[int, int]:
    """
    Calculate the total pixel dimensions of the image based on group count and column size.
    
    Accounts for external padding and vertical dividers.
    """
    column_width, column_height = column_size
    total_width = group_count * column_width # Base width from columns
    divider_count = max(0, group_count - 1)
    if vert_padding > 0:
        # If padding is greater than 0, the dividers function as an outline
        # for the individual columns, meaning there is one extra divider
        # per column.
        divider_count += group_count + 1 # Double then add two for the edges
    total_width += divider_count * vert_divider_width
    if vert_padding > 0:
        # Add padding before and after the image
        total_width += 2 * vert_padding
        # As well as between each column
        total_width += (group_count - 1) * vert_padding
    total_height = column_height
    return total_width, total_height

test_cmap = "hot"
# Colors:
# Datapoint: Mild red
# Slot: Mild green
# Column: Mild blue
# Group: Mild yellow
# Background: Gray
colors = {
    "datapoint": (230, 128, 128),
    "slot": (110, 150, 110),
    "column": (128, 128, 230),
    "group": (230, 230, 128),
    "background": (100, 100, 100),
}
def draw_image_frame(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    group_count: int,
    group_size: int,
    cell_size: int = 4,
    width_grid: int = 2,
    height_grid: int = 1,
    horiz_divider_width: int = 1,
    vert_divider_width: int = 1,
    vert_padding: int = 6,
) -> Tuple[Image.Image, ImageDraw.ImageDraw, List[List[Tuple[int, int, int, int]]]]:
    """
    Draw the frame of the image based on the provided parameters.
    0. Draw the background
    1. Draw the group dividers
    2. Draw the column dividers
    3. Draw the slot dividers
    4. Draw the cells
    5. Return the image and draw objects
    6. Return the list of cell bounding boxes by column
    """
    slot_size = calc_slot_size(
        cell_size=cell_size,
        width_grid=width_grid,
        height_grid=height_grid,
        horiz_divider_width=horiz_divider_width,
        vert_divider_width=vert_divider_width,
    )
    column_size = calc_column_size(
        group_size=group_size,
        slot_size=slot_size,
        horiz_divider_width=horiz_divider_width,
    )
    total_image_size = calc_total_image_size(
        group_count=group_count,
        column_size=column_size,
        vert_divider_width=vert_divider_width,
        vert_padding=vert_padding,
    )
    
    do_col_vert_outline = vert_padding > 0
    
    cell_bboxes: List[List[Tuple[int, int, int, int]]] = []
    
    initial_x = vert_padding
    current_x = initial_x
    for gx in range(group_count):
        # Draw the outline for the column
        col_x_start = current_x
        if do_col_vert_outline:
            # Group dividers
            if vert_divider_width > 1:
                # Draw the line as a rectangle to guarantee correct width
                draw.rectangle(
                    [
                        (col_x_start, 0),
                        (col_x_start + vert_divider_width - 1, 0),
                        (col_x_start + vert_divider_width - 1, total_image_size[1]),
                        (col_x_start, total_image_size[1]),
                    ],
                    fill=colors["slot"],
                )
            else:
                draw.line([(col_x_start, 0), (col_x_start, total_image_size[1])], fill=colors["slot"])
            current_x += vert_divider_width
        # Draw the column contents
        column_bboxes: List[Tuple[int, int, int, int]] = []
        current_y = 0
        for gy in range(group_size):
            slot_x_start = current_x
            slot_y_start = current_y
            # Draw the slot
            
            # Draw the cells
            cell_y_start = slot_y_start
            for hy in range(height_grid):
                cell_x_start = slot_x_start
                for wx in range(width_grid):
                    cell_x0 = cell_x_start
                    cell_y0 = cell_y_start
                    cell_x1 = cell_x0 + cell_size - 1
                    cell_y1 = cell_y0 + cell_size - 1
                    draw.rectangle(
                        [(cell_x0, cell_y0), (cell_x1, cell_y1)],
                        fill=colors["datapoint"],
                    )
                    column_bboxes.append((cell_x0, cell_y0, cell_x1, cell_y1))
                    cell_x_start += cell_size
                    if wx < width_grid - 1:
                        cell_x_start += vert_divider_width
                cell_y_start += cell_size
                if hy < height_grid - 1:
                    cell_y_start += horiz_divider_width
            # Move to the next slot
            current_y += slot_size[1]
            if gy < group_size - 1:
                current_y += horiz_divider_width
                if horiz_divider_width > 1:
                    # Draw the line as a rectangle to guarantee correct width
                    draw.rectangle(
                        [
                            (slot_x_start, current_y - horiz_divider_width),
                            (slot_x_start + slot_size[0] - 1, current_y - horiz_divider_width),
                            (slot_x_start + slot_size[0] - 1, current_y - 1),
                            (slot_x_start, current_y - 1),
                        ],
                        fill=colors["slot"],
                    )
                else:
                    draw.line(
                        [(slot_x_start, current_y - horiz_divider_width), (slot_x_start + slot_size[0] - 1, current_y - horiz_divider_width)],
                        fill=colors["slot"],
                    )
            pass
        current_x += column_size[0]
        # Draw the outline for the column
        if do_col_vert_outline or gx < group_count - 1:
            if vert_divider_width > 1:
                # Draw the line as a rectangle to guarantee correct width
                draw.rectangle(
                    [
                        (current_x, 0),
                        (current_x + vert_divider_width - 1, 0),
                        (current_x + vert_divider_width - 1, total_image_size[1]),
                        (current_x, total_image_size[1]),
                    ],
                    fill=colors["slot"],
                )
            else:
                draw.line([(current_x, 0), (current_x, total_image_size[1])], fill=colors["slot"])
            current_x += vert_divider_width
            current_x += vert_padding
        cell_bboxes.append(column_bboxes)
    return img, draw, cell_bboxes

def draw_example_image(
    group_count: int,
    group_size: int,
    cell_size: int = 4,
    width_grid: int = 2,
    height_grid: int = 1,
    horiz_divider_width: int = 1,
    vert_divider_width: int = 1,
    vert_padding: int = 6,
    upscale: int = 1,
) -> Image.Image:
    """
    Draw an example image based on the provided parameters.
    """
    slot_size = calc_slot_size(
        cell_size=cell_size,
        width_grid=width_grid,
        height_grid=height_grid,
        horiz_divider_width=horiz_divider_width,
        vert_divider_width=vert_divider_width,
    )
    column_size = calc_column_size(
        group_size=group_size,
        slot_size=slot_size,
        horiz_divider_width=horiz_divider_width,
    )
    total_image_size = calc_total_image_size(
        group_count=group_count,
        column_size=column_size,
        vert_divider_width=vert_divider_width,
        vert_padding=vert_padding,
    )
    img = Image.new("RGB", total_image_size, color=colors["background"])
    draw = ImageDraw.Draw(img)
    img, draw, cell_bboxes = draw_image_frame(
        img,
        draw,
        group_count=group_count,
        group_size=group_size,
        cell_size=cell_size,
        width_grid=width_grid,
        height_grid=height_grid,
        horiz_divider_width=horiz_divider_width,
        vert_divider_width=vert_divider_width,
        vert_padding=vert_padding,
    )
    if upscale > 1:
        img = img.resize((total_image_size[0] * upscale, total_image_size[1] * upscale), resample=Image.NEAREST)
    return img

def draw_serieses_to_image_frame_column(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    column_index: int,
    cell_bboxes: List[List[Tuple[int, int, int, int]]],
    serieses: List[List[np.number]],
    color_mapping: Callable[[float], Tuple[int, int, int]],
) -> Image.Image:
    """
    Draw the provided serieses into the provided image frame.
    
    Assumes that the image frame has already been drawn and that the cell bounding boxes
    are correct.
    
    Assumes the number of serieses corresponds to the number of cells per slot.
    """
    t0 = time.perf_counter()
    for slot_index, slot_series in enumerate(serieses):
        for index, value in enumerate(slot_series):
            cell_index = index * len(serieses) + slot_index
            cell_bbox = cell_bboxes[column_index][cell_index]
            cell_color = color_mapping(value)
            draw.rectangle(
                [(cell_bbox[0], cell_bbox[1]), (cell_bbox[2], cell_bbox[3])],
                fill=cell_color,
            )
    t1 = time.perf_counter()
    dt = t1 - t0
    if dt > 1.0:
        per_series = dt / len(serieses)
        per_cell = dt / sum(len(s) for s in serieses)
        print(f"Drew column {column_index} with {len(serieses)} series in {dt:.4f} s ({per_series:.6f} s/series, {per_cell:.6f} s/cell)")
    return img
    
if __name__ == '__main__':
    # Example usage and testing of the functions
    # Pull data from ~\NGIAB-CloudInfra\temp\AWI_16_2853886_006\forcings\by_catchment
    def get_data(folder_path: Path, total: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, pd.DataFrame]:
        if total is None:
            total = {}
        for item in os.listdir(folder_path):
            item = os.path.join(folder_path, item)
            if os.path.isfile(item):
                if item not in total:
                    total[Path(item).stem] = pd.read_csv(item)
        return total
    target_dir = Path.home() / "NGIAB-CloudInfra-old" / "temp" / "AWI_16_2853886_006" / "forcings" / "by_catchment"
    output_data = get_data(target_dir, {})
    group_count = len(output_data)
    group_size = len(next(iter(output_data.values())))
    print(f"{group_count=}, {group_size=}")
    slot_size = calc_slot_size()
    column_size = calc_column_size(group_size, slot_size)
    total_image_size = calc_total_image_size(group_count, column_size)
    print(f"{slot_size=}, {column_size=}, {total_image_size=}")
    
    # slot_size=(9, 4), column_size=(9, 3484), total_image_size=(3454, 3484)
    
    # Somewhat close to the original output size of 3908 x 4184?
    # Very uncertain where the extra space is coming from...
    
    def try_with_args(group_count, group_size, **kwargs):
        slot_size = calc_slot_size(**kwargs)
        column_size = calc_column_size(group_size, slot_size, **kwargs)
        total_image_size = calc_total_image_size(group_count, column_size, **kwargs)
        print(f"With {kwargs}:")
        print(f"  {slot_size=}, {column_size=}, {total_image_size=}")
        
    # Try increasing divider sizes
    try_with_args(group_count, group_size, vert_divider_width=2, horiz_divider_width=2)
    # slot_size=(10, 4), column_size=(10, 4180), total_image_size=(4144, 4180)
    # Much closer in height, but width is now too large..
    
    # Try increasing padding from 4 to 6, while dropping vert divider back to 1
    try_with_args(group_count, group_size, horiz_divider_width=2, vert_padding=6)
    # slot_size=(9, 4), column_size=(9, 4180), total_image_size=(3916, 4180)
    # Almost exact! Remaining inaccuracy can be chalked up to methodological differences.
    
    # Draw an example image
    example_img = draw_example_image(
        group_count=5,
        group_size=10,
        # vert_padding=6,
        upscale=10,
    )
    # example_img.show()
    
    # # Draw an example image matching the actual data
    # data_img = draw_example_image(
    #     group_count=group_count,
    #     group_size=group_size,
    # )
    # # data_img.show()
    
    # Draw an image frame for the actual data
    data_img = Image.new("RGB", total_image_size, color=colors["background"])
    data_draw = ImageDraw.Draw(data_img)
    data_img, data_draw, data_cell_bboxes = draw_image_frame(
        data_img,
        data_draw,
        group_count=group_count,
        group_size=group_size,
    )
    # data_series = []
    data_series: List[np.ndarray] = []
    for catchment_name, df in output_data.items():
        series = df['precip_rate'].to_numpy()
        data_series.append(series)
    # Create a color mapping function
    cmap = cm.get_cmap(test_cmap)
    norm = mcolors.Normalize(vmin=0, vmax=max(np.max(s) for s in data_series))
    def color_mapping(value: float) -> Tuple[int, int, int]:
        rgba = cmap(norm(value))
        return tuple(int(c * 255) for c in rgba[:3])
    # Draw the data into the image frame
    print("Drawing data into image frame...")
    t0 = time.perf_counter()
    last_time = t0
    last_index = 0
    for col_index in range(group_count):
        col_series = [data_series[col_index], data_series[col_index]]  # Assuming two cells per slot for simplicity
        data_img = draw_serieses_to_image_frame_column(
            data_img,
            data_draw,
            col_index,
            data_cell_bboxes,
            col_series,
            color_mapping,
        )
        current_time = time.perf_counter()
        dt = current_time - last_time
        if dt > 1.0:
            num_completed = col_index - last_index + 1
            per_col = dt / num_completed
            est_remaining = per_col * (group_count - col_index - 1)
            print(f"  Drew columns {last_index} to {col_index} ({num_completed}) in {dt:.4f} s ({per_col:.4f} s/col), estimated remaining time: {est_remaining:.2f} s")
            last_time = current_time
            last_index = col_index + 1
    t1 = time.perf_counter()
    if last_index < group_count:
        dt = t1 - last_time
        num_completed = group_count - last_index
        per_col = dt / num_completed
        print(f"  Drew columns {last_index} to {group_count - 1} ({num_completed}) in {dt:.4f} s ({per_col:.4f} s/col)")
    total_dt = t1 - t0
    print(f"Drew all {group_count} columns in {total_dt:.4f} s")
    # Previously: 14.5290 s for 229 columns
    # data_img.show()
    if not os.path.exists("dist"):
        os.makedirs("dist")
    data_img.save("dist/ngen_output_example.png")
    print("Saved example image to dist/ngen_output_example.png")
    
    # Demonstrate ability to compare different datasets
    # by making shifted versions of the data
    def make_shifted_data(data: List[np.ndarray], shift_amount: int) -> List[np.ndarray]:
        if shift_amount == 0:
            return [s.copy() for s in data]
        shifted_data = []
        for series in data:
            shifted_series = np.roll(series, shift_amount)
            shifted_data.append(shifted_series)
        return shifted_data
    data_serieses = []
    shifts = [0, 5, 10, 20]
    for shift in shifts:
        shifted_data = make_shifted_data(data_series, shift)
        data_serieses.append(shifted_data)
    # Create a larger image to hold all the datasets
    # (Done by increasing the width grid to match the number of datasets)
    comparison_group_count = group_count
    comparison_kwargs = { # Enumerate only the changed parameters
        "width_grid": len(shifts),
        "horiz_divider_width": 0,
        "vert_divider_width": 0,
        "vert_padding": 1,
    }
    comparison_slot_size = calc_slot_size(**comparison_kwargs)
    
    comparison_column_size = calc_column_size(
        group_size=group_size,
        slot_size=comparison_slot_size,
        **comparison_kwargs,
    )
    comparison_total_image_size = calc_total_image_size(
        group_count=comparison_group_count,
        column_size=comparison_column_size,
        **comparison_kwargs,
    )
    comparison_img = Image.new("RGB", comparison_total_image_size, color=colors["background"])
    comparison_draw = ImageDraw.Draw(comparison_img)
    comparison_img, comparison_draw, comparison_cell_bboxes = draw_image_frame(
        comparison_img,
        comparison_draw,
        group_count=comparison_group_count,
        group_size=group_size,
        **comparison_kwargs,
    )
    # Draw each dataset into the comparison image
    print("Drawing comparison data into image frame...")
    t2 = time.perf_counter()
    last_time = t2
    last_index = 0
    for col_index in range(comparison_group_count):
        col_serieses = []
        for shift_index in range(len(shifts)):
            col_serieses.append(data_serieses[shift_index][col_index])
        comparison_img = draw_serieses_to_image_frame_column(
            comparison_img,
            comparison_draw,
            col_index,
            comparison_cell_bboxes,
            col_serieses,
            color_mapping,
        )
        current_time = time.perf_counter()
        dt = current_time - last_time
        if dt > 1.0:
            num_completed = col_index - last_index + 1
            per_col = dt / num_completed
            est_remaining = per_col * (comparison_group_count - col_index - 1)
            print(f"  Drew columns {last_index} to {col_index} ({num_completed}) in {dt:.4f} s ({per_col:.4f} s/col), estimated remaining time: {est_remaining:.2f} s")
            last_time = current_time
            last_index = col_index + 1
    t3 = time.perf_counter()
    total_dt = t3 - t2
    if last_index < comparison_group_count:
        dt = t3 - last_time
        num_completed = comparison_group_count - last_index
        per_col = dt / num_completed
        print(f"  Drew columns {last_index} to {comparison_group_count - 1} ({num_completed}) in {dt:.4f} s ({per_col:.4f} s/col)")
    print(f"Drew all {comparison_group_count} columns in {total_dt:.4f} s")
    # Previously: 27.9437 s for 229 columns with 4 datasets
    # comparison_img.show()
    comparison_img.save("dist/ngen_output_comparison_example.png")
    print("Saved comparison image to dist/ngen_output_comparison_example.png")