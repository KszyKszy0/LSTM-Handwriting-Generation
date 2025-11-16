"""
Enhanced data augmentation script for your handwriting SVG dataset.

What this does:
- Loads your folders (the same structure you described).
- Computes character frequencies and an "importance" score for each line (rare sequences get higher importance).
- Picks the top-N most important (rare) lines and finds the corresponding SVG(s).
- For each selected SVG it creates several augmented copies using varied transforms (affine, jitter, elastic-like smoothing, stroke-width changes).
- Saves augmented SVGs into an augmentation output folder you specify and appends the labels to that folder's files.txt so the dataset stays consistent.

How to configure / run:
- Edit the `FOLDER_PATHS` list if needed.
- Edit AUGMENTATION_FOLDER to where you want augmented svgs saved (it will create folder if missing).
- Adjust TOP_K_RARE, AUG_PER_SAMPLE and AUG_METHODS to control how many rare words and augmentations are produced.

Notes / assumptions:
- The script expects each folder to have a files.txt with one label per SVG and SVG filenames that are numeric (e.g. 1.svg, 2.svg). It will also handle non-numeric basenames but numeric naming for augmented outputs is used to keep consistency.
- The augmentation methods are intentionally simple, deterministic enough to reproduce but random enough to generate variety. You can extend them.

"""

import os
import math
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
import shutil

# -------------------------- CONFIG --------------------------
# Folders that contain original SVGs and files.txt (relative to this script)
FOLDER_PATHS = ["../data/output", "../data/mwoutput", "../data/poloutput", "../data/hibru"]
# Where to put augmented SVGs and files.txt (will be created if missing)
AUGMENTATION_FOLDER = "../data/augmented"  # change to wherever you want
# How many of the top-most "important" (rare) lines to augment
TOP_K_RARE = 500
# How many augmented copies per found rare sample
AUG_PER_SAMPLE = 5
# Random seed for reproducibility
RANDOM_SEED = 42

# Which augmentation methods to pick from (strings map to functions below)
AUG_METHODS = [
    "affine_jitter",
    "jitter_only",
    "elastic_like",
    "scale_rotate",
    "thin_thicken"
]

# ------------------------ END CONFIG ------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------- Helpers: IO --------------------------

def abs_path_from_script(rel_path):
    """Return an absolute path relative to this script file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, rel_path))


def ensure_folder(folder):
    os.makedirs(folder, exist_ok=True)


def list_svg_files(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith('.svg')]
    return files


# ---------------------- SVG Parsing -------------------------

def parse_svg_polylines(svg_path):
    """Parse an SVG and return list of polylines. Each polyline is a list of (x,y).
    Works with <polyline points="x,y x,y ..." /> elements. If polylines are stored
    in other shapes this will ignore them (but can be extended).
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()
    # try to detect namespace
    ns = {k: v for k, v in [item.split('}') if '}' in item else (None, None) for item in []]}
    # Rather than worrying about ns, we'll search by tag ending
    polylines = []
    for elem in root.iter():
        if elem.tag.lower().endswith('polyline'):
            points_str = elem.attrib.get('points', '').strip()
            if not points_str:
                continue
            pairs = points_str.split()
            pts = []
            for pair in pairs:
                if ',' not in pair:
                    continue
                x_str, y_str = pair.split(',')
                try:
                    x = float(x_str)
                    y = float(y_str)
                except Exception:
                    continue
                pts.append([x, y])
            if pts:
                polylines.append(pts)
    # If no polylines found, try to find <path> and sample its 'd' (basic support)
    if not polylines:
        for elem in root.iter():
            if elem.tag.lower().endswith('path'):
                d = elem.attrib.get('d', '')
                sampled = _sample_path_d_basic(d)
                if sampled:
                    polylines.append(sampled)
    return polylines


def _sample_path_d_basic(d_str):
    """Very small fallback: parse only absolute moveto M and L commands roughly.
    This is intentionally basic — for complex paths consider using svgpathtools.
    """
    tokens = d_str.replace(',', ' ').split()
    pts = []
    i = 0
    current_cmd = None
    while i < len(tokens):
        t = tokens[i]
        if t.isalpha():
            current_cmd = t
            i += 1
            continue
        # numeric token
        if current_cmd in ('M', 'L'):
            try:
                x = float(t)
                y = float(tokens[i+1])
                pts.append([x,y])
                i += 2
                continue
            except Exception:
                i += 1
                continue
        else:
            i += 1
    return pts


def write_svg(polylines, save_path, stroke_width=3, stroke_color='black'):
    """Writes a minimal SVG with the provided polylines.
    polylines: list of list of [x,y]
    """
    # compute bounds
    xs = []
    ys = []
    for pl in polylines:
        for x,y in pl:
            xs.append(x)
            ys.append(y)
    if xs and ys:
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
    else:
        minx = miny = 0
        maxx = maxy = 100

    width = maxx - minx if maxx - minx > 0 else 100
    height = maxy - miny if maxy - miny > 0 else 100

    # create root element
    svg = ET.Element('svg', xmlns='http://www.w3.org/2000/svg', version='1.1',
                     width=str(width), height=str(height),
                     viewBox=f"{minx} {miny} {width} {height}")

    for pl in polylines:
        points_attr = ' '.join(f"{x},{y}" for x,y in pl)
        attrib = {
            'points': points_attr,
            'fill': 'none',
            'stroke': stroke_color,
            'stroke-width': str(stroke_width),
            'stroke-linecap': 'round',
            'stroke-linejoin': 'round'
        }
        ET.SubElement(svg, 'polyline', attrib)

    tree = ET.ElementTree(svg)
    tree.write(save_path, encoding='utf-8', xml_declaration=True)


# -------------------- Augmentation primitives --------------------

def center_of_polylines(polylines):
    xs = [x for pl in polylines for x,y in pl]
    ys = [y for pl in polylines for x,y in pl]
    return (sum(xs)/len(xs), sum(ys)/len(ys)) if xs and ys else (0,0)


def affine_transform_polylines(polylines, scale=1.0, rotate_deg=0.0, translate=(0,0)):
    theta = math.radians(rotate_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    cx, cy = center_of_polylines(polylines)
    out = []
    for pl in polylines:
        new_pl = []
        for x,y in pl:
            # translate to center
            x0 = x - cx
            y0 = y - cy
            # scale
            x1 = x0 * scale
            y1 = y0 * scale
            # rotate
            xr = x1 * cos_t - y1 * sin_t
            yr = x1 * sin_t + y1 * cos_t
            # translate back and apply additional translate
            xn = xr + cx + translate[0]
            yn = yr + cy + translate[1]
            new_pl.append([xn, yn])
        out.append(new_pl)
    return out


def jitter_polylines(polylines, sigma=0.8):
    out = []
    for pl in polylines:
        new_pl = [[x + random.gauss(0, sigma), y + random.gauss(0, sigma)] for x,y in pl]
        out.append(new_pl)
    return out


def elastic_like_polylines(polylines, sigma=2.5, alpha=1.0):
    # Very simple elastic-like: create random offsets and smooth them by neighbor averaging
    out = []
    for pl in polylines:
        n = len(pl)
        if n == 0:
            out.append([])
            continue
        # random offsets
        ox = np.random.randn(n) * sigma
        oy = np.random.randn(n) * sigma
        # smooth offsets: simple running average
        window = max(1, int(n * 0.05))
        def smooth(arr):
            res = np.copy(arr)
            for i in range(n):
                lo = max(0, i-window)
                hi = min(n, i+window+1)
                res[i] = np.mean(arr[lo:hi])
            return res
        ox_s = smooth(ox)
        oy_s = smooth(oy)
        new_pl = [[x + alpha*ox_s[i], y + alpha*oy_s[i]] for i,(x,y) in enumerate(pl)]
        out.append(new_pl)
    return out


def scale_stroke_polylines(polylines, factor=1.0):
    # leaves coordinates unchanged but we will return 'factor' so caller can set stroke-width
    return polylines, factor


def thin_thicken_polylines(polylines, factor=1.0):
    # small local inward/outward perturbation by moving points towards or away from center
    cx, cy = center_of_polylines(polylines)
    out = []
    for pl in polylines:
        new_pl = []
        for x,y in pl:
            vx = x - cx
            vy = y - cy
            new_pl.append([x + vx * (factor-1)*0.05, y + vy * (factor-1)*0.05])
        out.append(new_pl)
    return out


# -------------------- Utility: filename management --------------------

def next_numeric_filename(folder):
    """Finds next available numeric filename (k.svg). If no numeric names found returns 1.svg."""
    svgs = list_svg_files(folder)
    nums = []
    for s in svgs:
        name = os.path.splitext(s)[0]
        if name.isdigit():
            nums.append(int(name))
    if not nums:
        return 1
    return max(nums) + 1


def append_label_to_files_txt(folder, label):
    files_txt = os.path.join(folder, 'files.txt')
    with open(files_txt, 'a', encoding='utf-8') as f:
        f.write(label.rstrip('\n') + '\n')


# -------------------- Main augmentation flow --------------------

def load_dataset(folder_paths):
    """Returns list of dicts: [{folder, svg_files(list of basenames), files_txt_path, labels(list)}]"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    res = []
    for rel in folder_paths:
        folder = os.path.abspath(os.path.join(script_dir, rel))
        if not os.path.isdir(folder):
            print(f"Warning: folder not found: {folder}")
            continue
        files_txt = os.path.join(folder, 'files.txt')
        if not os.path.isfile(files_txt):
            print(f"Warning: files.txt not found in {folder}; skipping")
            continue
        with open(files_txt, 'r', encoding='utf-8') as f:
            labels = [line.rstrip('\n') for line in f]
        # list svg files and sort numerically when possible
        svgs = list_svg_files(folder)
        def keyfn(n):
            base = os.path.splitext(n)[0]
            return int(base) if base.isdigit() else base
        try:
            svgs.sort(key=keyfn)
        except Exception:
            svgs.sort()
        res.append({'folder': folder, 'svg_files': svgs, 'files_txt': files_txt, 'labels': labels})
    return res


def compute_char_stats(dataset_entries):
    counts = defaultdict(int)
    for entry in dataset_entries:
        for line in entry['labels']:
            for ch in line:
                counts[ch] += 1
    return counts


def reciprocal_weights(counts):
    recip = {}
    for k,v in counts.items():
        recip[k] = 1.0 / v if v > 0 else 0.0
    return recip


def importance_of_line(line, recip):
    # multiply reciprocals for each character (preserve duplicates). ignore newlines
    prod = 1.0
    for ch in line:
        if ch == '\n':
            continue
        prod *= recip.get(ch, 0.0)
    return prod


def find_top_k_rare_lines(dataset_entries, top_k=100):
    recip = reciprocal_weights(compute_char_stats(dataset_entries))
    importance_map = {}
    for entry in dataset_entries:
        for idx, line in enumerate(entry['labels']):
            if(len(line) < 15):
                continue
            importance_map[(entry['folder'], idx)] = importance_of_line(line, recip)
    # sort by importance descending (rare = larger product)
    sorted_items = sorted(importance_map.items(), key=lambda kv: -kv[1])
    return sorted_items[:top_k], recip


def augment_sample_and_save(entry, idx, aug_folder, aug_per_sample=5):
    """Given dataset entry dict and index of label/svg in it, create augmentations and save."""
    folder = entry['folder']
    label = entry['labels'][idx]
    svg_basename = entry['svg_files'][idx] if idx < len(entry['svg_files']) else None
    if svg_basename is None:
        print(f"No svg basename for {folder} index {idx}")
        return
    svg_path = os.path.join(folder, svg_basename)
    polylines = parse_svg_polylines(svg_path)
    if not polylines:
        print(f"No polylines parsed from {svg_path}; skipping")
        return

    ensure_folder(aug_folder)
    # determine starting numeric name
    # We will keep numeric naming scheme for augmented folder
    next_idx = next_numeric_filename(aug_folder)

    for a in range(aug_per_sample):
        method = random.choice(AUG_METHODS)
        stroke_w = 3.0
        pl_copy = [ [p[:] for p in pl] for pl in polylines ]

        if method == 'affine_jitter':
            s = random.uniform(0.9, 1.15)
            r = random.uniform(-3, 3)
            tx = random.uniform(-1, 1)
            ty = random.uniform(-1, 1)
            pl_copy = affine_transform_polylines(pl_copy, scale=s, rotate_deg=r, translate=(tx,ty))
            pl_copy = jitter_polylines(pl_copy, sigma=0.6)
            stroke_w = random.uniform(2.0, 4.0)

        elif method == 'jitter_only':
            pl_copy = jitter_polylines(pl_copy, sigma=random.uniform(0.6, 1.6))
            stroke_w = random.uniform(1.5, 3.5)

        elif method == 'elastic_like':
            pl_copy = elastic_like_polylines(pl_copy, sigma=random.uniform(1.0,3.5), alpha=random.uniform(0.6,1.6))
            pl_copy = jitter_polylines(pl_copy, sigma=0.5)
            stroke_w = random.uniform(2.0,4.5)

        elif method == 'scale_rotate':
            s = random.uniform(0.85, 1.25)
            r = random.uniform(-3, 3)
            pl_copy = affine_transform_polylines(pl_copy, scale=s, rotate_deg=r, translate=(0,0))
            stroke_w = random.uniform(1.8,3.8)

        elif method == 'thin_thicken':
            factor = random.uniform(0.7, 1.6)
            pl_copy = thin_thicken_polylines(pl_copy, factor=factor)
            pl_copy = jitter_polylines(pl_copy, sigma=0.4)
            stroke_w = random.uniform(1.2,4.8)

        # minor rescaling / normalization so that coordinates don't drift too large/small
        # center and scale to original bounding box
        # (We could reuse original bbox but keep coords untouched here)

        # Save augmented svg under numeric name
        save_name = f"{next_idx}.svg"
        save_path = os.path.join(aug_folder, save_name)
        write_svg(pl_copy, save_path, stroke_width=stroke_w)
        append_label_to_files_txt(aug_folder, label)
        next_idx += 1

    print(f"Augmented {svg_basename} from {os.path.basename(folder)} -> {aug_per_sample} samples in {aug_folder}")


def main():
    ensure_folder(abs_path_from_script(AUGMENTATION_FOLDER))
    dataset = load_dataset(FOLDER_PATHS)
    if not dataset:
        print("No valid dataset entries found. Check FOLDER_PATHS and presence of files.txt")
        return

    # compute top-K rare lines
    top_k, recip = find_top_k_rare_lines(dataset, top_k=TOP_K_RARE)
    print(f"Found {len(top_k)} most-important (rare) lines to augment")

    # For each rare line, find which dataset entry and index and augment
    for (folder_key, idx), importance in top_k:
        # find dataset entry
        entry = next((e for e in dataset if e['folder'] == folder_key), None)
        if entry is None:
            continue
        # call augmentation
        try:
            augment_sample_and_save(entry, idx, abs_path_from_script(AUGMENTATION_FOLDER), aug_per_sample=AUG_PER_SAMPLE)
        except Exception as exc:
            print(f"Error augmenting {folder_key} idx {idx}: {exc}")

    print("Done augmenting. Check the augmentation folder and files.txt")


if __name__ == '__main__':
    main()
