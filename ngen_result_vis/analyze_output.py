import os, sys
import pandas as pd
import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt

# this_dir=$(dirname $0)
# # temp/AWI_16_2853886_006/outputs
# output_dir="${this_dir}/temp/AWI_16_2853886_006/outputs"
this_dir = Path(sys.argv[0]).parent
print(f"this_dir: {this_dir}")
output_dir = this_dir / "temp/AWI_16_2853886_006/outputs/ngen/"
print(f"target_dir: {output_dir}, exists: {output_dir.exists()}")
# forcing_dir="${this_dir}/temp/AWI_16_2853886_006/forcings/by_catchment"
forcing_dir = this_dir / "temp/AWI_16_2853886_006/forcings/by_catchment"
print(f"forcing_dir: {forcing_dir}, exists: {forcing_dir.exists()}")

def get_data(folder_path, total={}):
    for item in os.listdir(folder_path):
        item = os.path.join(folder_path, item)
        if os.path.isfile(item):
            if item not in total:
                total[Path(item).stem] = pd.read_csv(item)
    return total

def limited_print(items, limit=10):
    out = ""
    for i, item in enumerate(items):
        if i >= limit:
            out += "..."
            break
        out += f"{item}\n"
    print(out)

output_data = get_data(output_dir, {})
print(f"output_data: {len(output_data)}")
# limited_print(output_data)
forcing_data = get_data(forcing_dir, {})
print(f"forcing_data: {len(forcing_data)}")

# prepare comparison
def rho(temp):
    """
        Calculate water density at temperature
    """
    temp = temp - 273.15
    return 999.99399 + 0.04216485*temp - 0.007097451*(temp**2) + 0.00003509571*(temp**3) - 9.9037785E-8*(temp**4)


# "partial_calc__TMP_2maboveground_apply_rho",
# "partial_calc__APCP_surface_shifted",
# "partial_calc__shifted_div_rho"

def aorc_as_rate(dataFrame: pd.DataFrame):
    """
        Convert kg/m^2 -> m/s
    """
    if "APCP_surface" not in dataFrame.columns:
        raise KeyError("APCP_surface not in dataFrame columns")
    if isinstance(dataFrame.index, pd.MultiIndex):
        interval = pd.Series(dataFrame.index.get_level_values(0))
    else:
        interval = pd.Series(dataFrame.index)
    interval = ( interval.shift(-1) - interval ) / np.timedelta64(1, 's').astype(float)
    interval.index = dataFrame.index
    # precip_rate = ( dataFrame['APCP_surface'].shift(-1) / dataFrame['TMP_2maboveground'].apply(rho) ) / interval
    try:
        partial_calc__TMP_2maboveground_apply_rho = dataFrame['TMP_2maboveground'].apply(rho)
    except KeyError as e:
        print(f"KeyError: TMP_2maboveground not found in dataFrame. Columns: {dataFrame.columns}")
        raise e
    try:
        partial_calc__APCP_surface_shifted = dataFrame['APCP_surface'].shift(-1)
    except KeyError as e:
        print(f"KeyError: APCP_surface not found in dataFrame. Columns: {dataFrame.columns}")
        raise e
    partial_calc__shifted_div_rho = partial_calc__APCP_surface_shifted / partial_calc__TMP_2maboveground_apply_rho
    precip_rate = partial_calc__shifted_div_rho / interval
    calc_df = pd.DataFrame({
        'precip_rate': precip_rate,
        'interval': interval,
        'partial_calc__TMP_2maboveground_apply_rho': partial_calc__TMP_2maboveground_apply_rho,
        'partial_calc__APCP_surface_shifted': partial_calc__APCP_surface_shifted,
        'partial_calc__shifted_div_rho': partial_calc__shifted_div_rho
    })
    return calc_df

# calc_data = {}

# for filename, df in forcing_data.items():
#     try:
#         calc_data[filename] = aorc_as_rate(df)
#     except KeyError as e:
#         print(f"KeyError: in {filename}")
#         raise e

# delta_df = {}
# means = {}
# manual_means = {} #ignore zero values
# for filename, df in calc_data.items():
#     df: pd.DataFrame
#     delta_df[filename] = df.copy()
#     delta_df[filename].fillna(0, inplace=True)
#     delta_df[filename]['precip_rate_diff'] = output_data[filename]['atmosphere_water__precipitation_rate'] * 3600 - df['precip_rate']
#     for col in df.columns:
#         if col not in output_data[filename].columns:
#             continue
#         delta_df[filename][col + "_diff"] = output_data[filename][col]
#     delta_df: dict[str, pd.DataFrame]
#     means[filename] = delta_df[filename].mean(skipna=True, numeric_only=True)
#     tmp_manual_means = pd.Series()
#     for col in delta_df[filename].columns:
#         if col.endswith("_diff"):
#             val = 0
#             num = 0
#             for v in delta_df[filename][col]:
#                 if v != 0 and not np.isnan(v):
#                     val += abs(v)
#                     num += 1
#             num = max(num, 1)
#             tmp_manual_means[col] = val / num
#     manual_means[filename] = tmp_manual_means



# for i, (filename, df) in enumerate(means.items()):
#     # print(f"means[{i}]: {filename}")
#     # print(df.head())
#     # print(f"Manual means[{i}]: {filename}")
#     # print(manual_means[filename].head())
#     if i > 10:
#         break

# compare APCP_surface_shifted to APCP_surface
shifted_diff = {}
manual_shift_means = {}
for filename, df in output_data.items():
    if not "cat" in filename:
        continue
    try:
        # shifted_diff[filename] = df['APCP_surface_shifted'] * 1000 - forcing_data[filename]['APCP_surface'].shift(1)
        # forcing_data doesn't contain an APCP_surface.... but the df does?
        # let's see what happens
        shifted_diff[filename] = df['APCP_surface_shifted'].shift(-1) * 1000 - df['APCP_surface'].shift(1)
    except KeyError as e:
        print(f"KeyError: in {filename}. Columns: {df.columns}")
        raise e
    manual_shift_means[filename] = 0
    val = 0
    num = 0
    for v in shifted_diff[filename]:
        if v != 0 and not np.isnan(v):
            val += abs(v)
            num += 1
    num = max(num, 1)
    manual_shift_means[filename] = val / num
    

# def make_subplots():
#     fig, ax = plt.subplots(3, 2)
#     ax = ax.flatten()
#     for i, (filename, df) in enumerate(delta_df.items()):
#         ax[0].plot(df.index, df['precip_rate_diff'], label=filename)
#         # ax[1].plot(df.index, df['interval_diff'], label=filename)
#         ax[1].plot(df.index, df['partial_calc__TMP_2maboveground_apply_rho_diff'], label=filename)
#         ax[2].plot(df.index, df['partial_calc__APCP_surface_shifted_diff'], label=filename)
#         ax[3].plot(df.index, df['partial_calc__shifted_div_rho_diff'], label=filename)
#         ax[4].plot(forcing_data[filename].index, shifted_diff[filename], label=filename)
#     ax[0].set_title("precip_rate_diff")
#     # ax[1].set_title("interval_diff")
#     ax[1].set_title("partial_calc__TMP_2maboveground_apply_rho_diff")
#     ax[2].set_title("partial_calc__APCP_surface_shifted_diff")
#     ax[3].set_title("partial_calc__shifted_div_rho_diff")
#     ax[4].set_title("APCP_surface_shifted - APCP_surface")
#     plt.show()
        
# make_subplots()

# Compare APCP_surface_shifted to APCP_surface
# How much are they shifted?
# How much are they different?

import PIL
from PIL import Image, ImageDraw, ImageFont

# Compare the two series by producing an image where the coloration is based on the percentile of the value
# in the two series

import matplotlib.colors as mcolors
import matplotlib.cm as cm

import math
import numpy as np

total_values = 0
total_values += len(shifted_diff)
total_values += len(forcing_data)
values_per_group = len(next(iter(forcing_data.values())))
total_values *= values_per_group
print(f"total_values: {total_values}")
print(f"Pairs: {total_values / 2}")
print(f"total groups: {len(forcing_data)}")

# #reasonable aspect ratio
# # approx 16:9 or 1280:720
# width = 1280
# height = 720

# width_per_group = width / len(forcing_data)
# width_per_group_2 = math.ceil(width_per_group)
# print(f"width per group: {width_per_group} -> {width_per_group_2}")

# adjusted_width = width_per_group_2 * len(forcing_data)
# print(f"adjusted_width: {adjusted_width}")

# total_per_group = values_per_group * 2
# rows_needed_per_group = (total_per_group / width_per_group_2)
# print(f"rows_needed_per_group: {rows_needed_per_group}")

def find_reasonable_aspect_ratio(group_size, group_count, width_grid=2, height_grid=1, resolution_mult=2):
    """
    Using the provided information, find a reasonable aspect ratio
    Mostly by taking the square root and then choosing the nearest common aspect ratio
    """
    total_cells = group_size * group_count
    total_pixels = total_cells * width_grid * height_grid * resolution_mult**2
    # find the square root
    cell_sqrt = math.sqrt(total_pixels)
    
    common_aspect_ratios = [
        (854, 480), # 16:9
        (1024, 576),
        (1280, 720),
        (1366, 768),
        (1600, 900),
        (1920, 1080),
        (2560, 1440),
        (3840, 2160),
        (7680, 4320),
        (15360, 8640)
    ]
    aspect_sqrts = [math.sqrt(x[0] * x[1]) for x in common_aspect_ratios]
    grouped_aspect = [(x, y) for x, y in zip(common_aspect_ratios, aspect_sqrts)]
    nearest_func = lambda x: abs(x[1] - cell_sqrt)
    grouped_aspect.sort(key=nearest_func)
    print(f"cell_sqrt: {cell_sqrt}")
    for i, (ratio, aspect) in enumerate(grouped_aspect):
        print(f"aspect: {aspect}, ratio: {ratio}")
    return grouped_aspect[0][0]

res=find_reasonable_aspect_ratio(values_per_group, len(forcing_data))
print(res)

def get_allocated_space(aspect_ratio, group_count, width_grid=2, height_grid=1, resolution_mult=4):
    width, height = aspect_ratio
    width_per_group = width / group_count / width_grid / resolution_mult
    width_per_group_2 = math.ceil(width_per_group)
    width_per_group_2 *= width_grid * resolution_mult
    adjusted_width = width_per_group_2 * group_count
    print(f"adjusted_width: {adjusted_width}, width_per_group: {width_per_group}, width_per_group_2: {width_per_group_2}")
    effective_columns = width_per_group_2 / width_grid / resolution_mult
    total_per_group = values_per_group
    total_per_column = total_per_group / effective_columns
    print(f"total_per_group: {total_per_group}, total_per_column: {total_per_column}")
    pixels_per_column = total_per_column * height_grid * resolution_mult
    print(f"pixels_per_column: {pixels_per_column}")
    max_per_column = height / height_grid / resolution_mult
    max_slots = max_per_column * effective_columns
    group_allocations = {}
    for i in range(group_count):
        group_allocations[i] = {
            "start_x": i * width_per_group_2,
            "end_x": (i + 1) * width_per_group_2,
            "start_y": 0,
            "end_y": height,
            "slots_per_column": round(max_per_column),
            "max_slots": round(max_slots),
            "cell_width": width_grid * resolution_mult,
            "cell_height": height_grid * resolution_mult,
            "pixel_width": resolution_mult,
            "pixel_height": resolution_mult,
            "columns": round(effective_columns),
            "rows": round(max_per_column)
        }
    return adjusted_width, height, group_allocations

adjusted_width, height, group_alloc = get_allocated_space(res, len(forcing_data))
print(f"adjusted_width: {adjusted_width}, height: {height}")
# print(group_alloc)

def group_ind_to_xy(group_ind, group_alloc):
    # Take the index of the cell and convert it to position data
    group = group_alloc
    x = group["start_x"]
    y = group["start_y"]
    if group_ind >= group["max_slots"]:
        raise ValueError(f"Index {group_ind} is out of range ({group['max_slots']})")
    col = group_ind // group["slots_per_column"]
    row = group_ind % group["slots_per_column"]
    x += col * group["cell_width"]
    y += row * group["cell_height"]
    return col, row, x, y

def matplotlib_cmap_to_pil(cmap: mcolors.Colormap, index, min_val=0, max_val=1)->tuple:
    if min_val != 0 or max_val != 1:
        index = (index - min_val) / (max_val - min_val)
        # index = math.sqrt(index)
    if index < 0:
        return (0, 0, 0)
    color = cmap(index)
    result = tuple(int(x * 255) for x in color[:3])
    # print(f"ind {index} -> {result}") 
    return result




def make_canvas(width, height, color=(255, 255, 255)):
    return Image.new('RGB', (width, height), color)

def test_image(width, height, group_allocs, cmap):
    canvas = make_canvas(width, height)
    draw = ImageDraw.Draw(canvas)
    first = group_allocs[0]
    slot_count = first["max_slots"]
    colormap = cm.get_cmap(cmap, lut=slot_count)
    slot_width = first["cell_width"]
    slot_height = first["cell_height"]
    for i, group_alloc in group_allocs.items():
        for j in range(slot_count):
            _,_,x, y = group_ind_to_xy(j, group_alloc)
            color = matplotlib_cmap_to_pil(colormap, j)
            draw.rectangle([x, y, x + slot_width, y + slot_height], fill=color)
    return canvas

test_cmap = "rainbow"
# test_img = test_image(adjusted_width, height, group_alloc, test_cmap)
# test_img.show()

def pad_bbox(bbox, padding)->tuple:
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    left, top, right, bottom = bbox
    left -= padding[0]
    top -= padding[1]
    right += padding[2]
    bottom += padding[3]
    left = max(0, left)
    top = max(0, top)
    return left, top, right, bottom

def bbox_contained_in(bbox1, bbox2):
    left1, top1, right1, bottom1 = bbox1
    left2, top2, right2, bottom2 = bbox2
    return left1 >= left2 and top1 >= top2 and right1 <= right2 and bottom1 <= bottom2

def necessary_bounds_for_padding(image_bbox, new_rect, padding, threshold_ratio=0.5)->tuple[bool, tuple]:
    """
    Given the bounding box of the image and a new rectangle, determine if the new rectangle
    is within the image and if not, return the necessary bounds to shift the image
    returns: (emergency, (left, top, right, bottom))
    Emergency is True if the new rectangle is outside the image, and is in danger of being cropped
    Otherwise, the returned tuple is a recommendation.
    """
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    left, top, right, bottom = new_rect
    img_left, img_top, img_right, img_bottom = image_bbox
    threshold_padding = [int(- x * threshold_ratio) if i > 1 else 0 for i, x in enumerate(padding)]
    threshold_bbox = pad_bbox(image_bbox, threshold_padding)
    emergency = not bbox_contained_in(new_rect, threshold_bbox)
    new_left = img_left
    new_top = img_top
    new_right = img_right
    new_bottom = img_bottom
    if left < img_left:
        new_left = img_left
    if top < img_top:
        new_top = img_top
    if right > img_right:
        new_right = img_right
    if bottom > img_bottom:
        new_bottom = img_bottom
    return emergency, pad_bbox((new_left, new_top, new_right, new_bottom), padding)



def all_min_max(data:dict[str, pd.DataFrame])->tuple:
    first = next(iter(data.values()))
    min_val = first.min().min()
    max_val = first.max().max()
    for key, df in data.items():
        min_val = min(min_val, df.min().min())
        max_val = max(max_val, df.max().max())
    return min_val, max_val

def series_comparisons_image(width, height, group_allocs, series1, series2, cmap):

    first = group_allocs[0]
    slot_count = first["max_slots"]
    all_min, all_max = all_min_max(series1)
    all_min2, all_max2 = all_min_max(series2)
    all_min = min(all_min, all_min2)
    all_max = max(all_max, all_max2)
    all_range = all_max - all_min
    colormap = cm.get_cmap(cmap)#, lut=int(all_range))
    heatmap = cm.get_cmap("hot")
    print(f"all_min: {all_min}, all_max: {all_max}")

    num_items = len(series1[next(iter(series1))])
    for i, group_alloc in group_allocs.items():
        group_allocs[i]["slots_per_column"] = math.ceil(num_items / 100) * 100
        print(f"slots_per_column: {group_allocs[i]['slots_per_column']}")
        


    # Dimension calculations
    datapoint_dims = (group_allocs[0]["pixel_width"], group_allocs[0]["pixel_height"])
    # Slot dimensions-
    # 2 datapoints per slot
    # 1 pixel padding between datapoints
    # 1 pixel gap around the slot to allow for the outline
    slot_dims = (datapoint_dims[0] * 2 + 4, datapoint_dims[1] + 2)
    # Column dimensions
    # group_alloc["slots_per_column"] slots per column
    # No padding between slots- their outlines will be touching
    # 1 pixel padding around the column to allow for the outline
    column_width = slot_dims[0] + 2
    column_height = slot_dims[1] * group_allocs[0]["slots_per_column"] + 2
    column_dims = (column_width, column_height)
    # Group dimensions
    # group_alloc["columns"] columns per group
    # 1 pixel padding between columns (column_count - 1 pixels)
    # 1 pixel padding around the group to allow for the outline (2 pixels for each dimension)
    # group_width = max(group_allocs[0]["columns"], 1) * column_width + (group_allocs[0]["columns"] - 1) + 2
    _group_width = lambda col_count: col_count * (column_width + 1) + (col_count)
    group_height = column_height + 2
    _group_dims = lambda col_count: (_group_width(col_count), group_height)
    # Drawing CommandQueue: Allows us to choose the correct size for the canvas after we calculate what we need
    draw_commandqueue = []

    bgcol = (100, 100, 100)
    canvas = make_canvas(width, height, bgcol)
    def check_bounds():
        nonlocal canvas
        _left, _top, _right, _bottom = canvas.getbbox()
        nonlocal width, height
        width = _right - _left
        height = _bottom - _top
        return width, height
    # arrange series into pair groups
    series_pairs = [] # filename, series1, series2
    cellwidths = {}
    for key, df in series1.items():
        series_pairs.append((key, df, series2[key]))

    used_width = 0
    for i, group_alloc in list(group_allocs.items()):
        series_pair = series_pairs[i]
        # groups divided vertically, iterate over them horizontally
        # Calculated values from the group_alloc are only correct for row and column stats.
        # For example, all of the start_x, start_y, end_x, end_y values are completely wrong
        # We recalculated everything else here

        num_cols = math.ceil(len(series_pair[1]) / group_alloc["slots_per_column"])
        group_dims = _group_dims(num_cols)
        group_width, group_height = group_dims
        used_width += group_width + 1

        num_per_column = []
        _used = 0
        for j in range(num_cols):
            _used += min(group_alloc["slots_per_column"], len(series_pair[1]) - _used)
            num_per_column.append(_used)

        

        def group_bbox():
            nonlocal i, group_dims
            x = used_width - group_dims[0]
            y = 0
            return (x, y, x + group_dims[0], y + group_dims[1])
        
        def column_bbox(col, group_bbox=group_bbox()):
            accessible_bbox = (group_bbox[0] + 1, group_bbox[1] + 1, group_bbox[2] - 1, group_bbox[3] - 1)
            x = accessible_bbox[0] + col * column_dims[0]
            y = accessible_bbox[1]
            return (x, y, x + column_dims[0], y + column_dims[1])
        
        def slot_bbox(row, column_bbox=column_bbox(0)):
            accessible_bbox = (column_bbox[0] + 1, column_bbox[1] + 1, column_bbox[2] - 1, column_bbox[3] - 1)
            x = accessible_bbox[0]
            y = accessible_bbox[1] + row * slot_dims[1]
            return (x, y, x + slot_dims[0], y + slot_dims[1])
        
        def datapoint_bbox(index, slot_bbox=slot_bbox(0)):
            accessible_bbox = (slot_bbox[0] + 1, slot_bbox[1] + 1, slot_bbox[2] - 1, slot_bbox[3] - 1)
            total_width = accessible_bbox[2] - accessible_bbox[0]
            x = accessible_bbox[0] if index == 0 else accessible_bbox[0] + total_width - datapoint_dims[0]
            y = accessible_bbox[1]
            return (x, y, x + datapoint_dims[0], y + datapoint_dims[1])

        def slot_ind_to_col_row(ind):
            col = ind // group_alloc["slots_per_column"]
            row = ind % group_alloc["slots_per_column"]
            return col, row
        
        def subdivide_indices_to_col_row(indices):
            # list or range -> 2d list of col, row
            result = []
            _i=0
            while _i < len(indices):
                new_col = []
                for _j in range(group_alloc["slots_per_column"]):
                    if _i >= len(indices):
                        break
                    new_col.append(indices[_i])
                    _i += 1
                result.append(new_col)
            return result
        
        def draw_bbox(bbox, **kwargs):
            nonlocal draw_commandqueue
            draw_commandqueue.append((bbox, kwargs))

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
            "background": bgcol
        }
        def get_subcomponent_bboxes():
            nonlocal group_alloc
            all_bboxes = {}
            all_bboxes["group"] = group_bbox()
            all_bboxes["columns"] = [column_bbox(_k, group_bbox=all_bboxes["group"]) for _k in range(num_cols)]
            all_bboxes["slots"] = [
                [
                    slot_bbox(j, column_bbox=all_bboxes["columns"][_k]) for j in range(num_per_column[_k])
                ] for _k in range(num_cols)
            ]
            all_bboxes["datapoints"] = [
                [
                    (
                        datapoint_bbox(0, slot_bbox=all_bboxes["slots"][_k][j]), 
                        datapoint_bbox(1, slot_bbox=all_bboxes["slots"][_k][j])
                     ) for j in range(num_per_column[_k])
                ] for _k in range(num_cols)
            ]
            return all_bboxes

        def draw_subcomponent_bboxes():
            nonlocal draw_bbox, series_pair, colormap, all_min, all_max
            all_bboxes = get_subcomponent_bboxes()
            # print(f"Acquired bboxes: len(col)={len(all_bboxes['columns'])}, len(slot)={len(all_bboxes['slots'][0])}, len(datapoints)={len(all_bboxes['datapoints'][0])}")
            # print(f"all_bboxes: {all_bboxes}")
            series1, series2 = series_pair[1], series_pair[2]
            # draw_bbox(all_bboxes["group"], outline=matplotlib_cmap_to_pil(colormap, 7*i%30, 0, 30))
            # for col, col_bbox in enumerate(all_bboxes["columns"]):
            #     draw_bbox(col_bbox, outline=colors["column"])
            _i=0
            # print("0")
            while _i < len(series1):
                # print(f"{_i}/{len(series1)}", end="\r")
                col, row = slot_ind_to_col_row(_i)
                try:
                    slot_bbox = all_bboxes["slots"][col][row]
                except IndexError as e:
                    print(f"IndexError: col={col}, row={row} for slot index {_i}")
                    print("len(series1):", len(series1))
                    raise e
                
                datapt_0_bbox, datapt_1_bbox = all_bboxes["datapoints"][col][row]
                #Check to ensure the pd.Series indices are non-NaN and non-zero
                val1 = series1.iloc[_i]
                val2 = series2.iloc[_i]
                if (np.isnan(val1) or val1 == 0) and (np.isnan(val2) or val2 == 0):
                    _i+=1
                    continue
                draw_bbox(slot_bbox, outline=colors["slot"])

                color1 = matplotlib_cmap_to_pil(heatmap, (val1 - all_min) / all_range)
                color2 = matplotlib_cmap_to_pil(heatmap, (val2 - all_min) / all_range)

                # Draw the datapoints
                draw_bbox(
                    datapt_0_bbox, 
                    # outline=colors["datapoint"], 
                    fill=color1
                    )
                draw_bbox(
                    datapt_1_bbox, 
                    # outline=colors["datapoint"], 
                    fill=color2
                    )

                _i += 1
            # print(f"Drawn {len(series1)} slots")



        draw_subcomponent_bboxes()
        # measured_dims = {
        #     "datapoint": slot_bbox(0, 0, 0),
        #     "slot": slot_bbox(0, 0),
        #     "column": column_bbox(0),
        #     "group": group_bbox()
        # }
        # wdth_hghts = {}
        # for key, bbox in measured_dims.items():
        #     left, top, right, bottom = bbox
        #     wdth_hghts[key] = (right - left, bottom - top)

        # print(f"measured_dims:")
        # for key, bbox in measured_dims.items():
        #     print(f"{key}: {bbox}")
        # print(f"wdth_hghts:")
        # for key, bbox in wdth_hghts.items():
        #     print(f"{key}: {bbox}")
        

            




    width, height = check_bounds()
    min_x = 0
    min_y = 0
    max_x = width
    max_y = height
    bbox_full = (min_x, min_y, max_x, max_y)
    def bbox_update(bbox0, bbox_new):
        # Update the bounding box to include the new one
        left0, top0, right0, bottom0 = bbox0
        left, top, right, bottom = bbox_new
        left0 = min(left0, left)
        top0 = min(top0, top)
        right0 = max(right0, right)
        bottom0 = max(bottom0, bottom)
        return left0, top0, right0, bottom0
    
    for bbox, kwargs in draw_commandqueue:
        bbox_full = bbox_update(bbox_full, bbox)
    
    print(f"bbox_full: {bbox_full}")
    print(f"canvas bbox: {canvas.getbbox()}")

    if not bbox_contained_in(bbox_full, canvas.getbbox()):
        print("Warning: Image is not contained in the canvas")
        print(f"Rebuilding canvas")
        new_width = bbox_full[2] - bbox_full[0]
        new_height = bbox_full[3] - bbox_full[1]
        canvas = make_canvas(new_width, new_height, bgcol)
        draw = ImageDraw.Draw(canvas)
        
    for bbox, kwargs in draw_commandqueue:
        draw.rectangle(bbox, **kwargs)

    print(cellwidths)
    # canvas = canvas.crop((0, height-300, 30, height))
    def focus_on_not_color(color, canvas):
        #given a background color, find a region that has other colors
        # and crop to focus on that region with a 10 pixel padding
        _left, _top, _right, _bottom = canvas.getbbox()
        nonlocal width, height
        width = _right - _left
        height = _bottom - _top
        left = _right
        right = _left
        top = _bottom
        bottom = _top
        print(f"mins: {left}, {top}, maxs: {right}, {bottom}")
        for x in range(_left, _right):
            for y in range(_top, _bottom):
                if canvas.getpixel((x, y)) != color:
                    left = min(left, x)
                    right = max(right, x)
                    top = min(top, y)
                    bottom = max(bottom, y)
                    # draw.point((x, y), fill=(255, 0, 0))
        print(f"mins: {left}, {top}, maxs: {right}, {bottom}")
        left = max(0, left - 10)
        right = min(width, right + 10)
        top = max(0, top - 10)
        bottom = min(height, bottom + 10)
        return canvas.crop((left, top, right, bottom))
    print(f"canvas bbox: {canvas.getbbox()}")
    canvas = focus_on_not_color(bgcol, canvas)
    print(f"canvas bbox: {canvas.getbbox()}")


            




    return canvas

apcp_shifted = {}

apcp = {}
all_min_1 = 0
all_min_2 = 0
all_max_1 = 0
all_max_2 = 0
# for key, df in forcing_data.items():
# forcing_data doesn't have APCP_surface, so use output_data
for key, df in output_data.items():
    if "cat" not in key:
        continue
    try:
        apcp_shifted[key] = output_data[key]['APCP_surface_shifted'].shift(-1) * 1000
    except KeyError as e:
        print(f"KeyError: in {key}. Columns: {df.columns}")
        raise e
    apcp[key] = df['APCP_surface']
    all_min_1 = min(all_min_1, apcp_shifted[key].min())
    all_min_2 = min(all_min_2, apcp[key].min())
    all_max_1 = max(all_max_1, apcp_shifted[key].max())
    all_max_2 = max(all_max_2, apcp[key].max())
print(f"APCP_surface_shifted: {all_min_1} - {all_max_1}")
print(f"APCP_surface: {all_min_2} - {all_max_2}")
# exit()
# zero or the minimum value should be black in the colormap
# higher values should be more intense colors

cmap_heatmap = "rainbow"
# cmap_heatmap = "hot"

cmp_img = series_comparisons_image(adjusted_width, height, group_alloc, apcp_shifted, apcp, cmap_heatmap)
cmp_img.save("comparison.png")


