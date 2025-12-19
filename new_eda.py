# species_top50_plus_other_barv_blue_labels_tight.py
# ------------------------------------------------------------
# Bar chart: top-50 species + final "Other (N spp.)" bar.
# - Blues binned gradient
# - Species ticks at 90°
# - Numeric labels anchored just above bar tops (offset in points)
# ------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib.patches as mpatches

# ---------------- Config ----------------
dataset_dir   = r"C:\Users\ADMIN\Downloads\snake_datasetNew"   # root with subfolders per species
save_basename = "species_top50_plus_other_barv_blue_labels_tight"

# Figure size (inches)
fig_width_in  = 12.0        # match your \textwidth for 2-col
fig_height_in = 6.0         # set to None to auto-scale with N
base_h_in     = 5.0         # used only if fig_height_in is None
per_label_h_in = 0.05       # extra height per species for tick labels

font_sz         = 8
val_rot_deg     = 0        # rotation for numeric value labels (set 0 for horizontal)
label_offset_pts = 0        # vertical offset of value labels ABOVE bar top, in points

# Keep top-K species; collapse the rest into "Other"
top_k_keep    = 50
label_other_with_count = True  # "Other (N spp.)"
min_value_for_label = 1        # skip printing tiny labels

# -------------- Helpers ---------------
image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
def is_image(fname: str) -> bool:
    return os.path.splitext(fname.lower())[1] in image_exts

def pretty_name(dirname: str) -> str:
    name = dirname.replace("_", " ").replace("-", " ").strip()
    return name if any(c.isupper() for c in name) else name.title()

# -------- Count images per species --------
class_counts = {}
with os.scandir(dataset_dir) as it:
    for entry in it:
        if entry.is_dir():
            species = pretty_name(entry.name)
            count = 0
            for root, _dirs, files in os.walk(entry.path):
                count += sum(1 for f in files if is_image(f))
            class_counts[species] = count

# Drop zeros, sort descending
items = [(s, n) for s, n in class_counts.items() if n > 0]
items.sort(key=lambda x: x[1], reverse=True)
if not items:
    raise SystemExit("No species with >0 images found.")

# Keep top-K, collapse the rest into 'Other'
if top_k_keep is not None and len(items) > top_k_keep:
    top_items = items[:top_k_keep]
    tail = items[top_k_keep:]
    other_count = sum(n for _, n in tail)
    other_species_n = len(tail)
    other_label = f"Other ({other_species_n} spp.)" if label_other_with_count else "Other"
    top_items.append((other_label, other_count))
else:
    top_items = items

species, counts = zip(*top_items)
N = len(species)
y_max = max(counts)

# -------- Figure size --------
if fig_width_in is None:
    fig_width_in = 14.0
if fig_height_in is None:
    fig_height_in = max(2.0, base_h_in + per_label_h_in * N)

fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))

# -------- Binned gradient colouring (Blues) --------
# bins: <100, 100–199, 200–299, 300–399, 400–499, ≥500 (last present only if y_max >= 500)
base_edges = [0, 100, 200, 300, 400, 500]
edges = [e for e in base_edges if e <= y_max] + [y_max + 1]
n_bins = len(edges) - 1
cmap = cm.get_cmap("Blues", n_bins)
norm = colors.BoundaryNorm(edges, ncolors=cmap.N, clip=True)
bar_colors = cmap(norm(counts))

# Legend labels
bin_labels = []
for i in range(n_bins):
    lo, hi_excl = edges[i], edges[i+1]
    hi = hi_excl - 1
    if i == 0 and lo == 0:
        bin_labels.append("<100" if hi >= 99 else f"0–{hi}")
    elif i == n_bins - 1 and edges[-2] >= 500:
        bin_labels.append("≥500")
    else:
        bin_labels.append(f"{lo}–{hi}")

# -------- Plot --------
x = np.arange(N)
bars = ax.bar(x, counts, color=bar_colors, align="center")

print(species, counts)

# Place ticks at actual bar centers to avoid misalignment
centers = [b.get_x() + b.get_width() / 2 for b in bars]
ax.set_xticks(centers)
ax.set_xticklabels(species, rotation=90, ha="center", va="top", fontsize=font_sz)

# Axis labels (no title for cleaner export)
ax.set_ylabel("Sample Count", fontsize=font_sz)
ax.set_xlabel("Species", fontsize=font_sz)

# Grid & spines
ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_linewidth(0.8)

# Tight x-limits; add headroom so labels don't clip
ax.set_xlim(-0.5, N - 0.5)
ax.set_ylim(0, y_max * 1.12)

# -------- Numeric value labels just above bar tops --------
# Use annotate with "offset points" so the label starts right after the bar edge,
# independent of data scale.
for b, ctr, v in zip(bars, centers, counts):
    if v < min_value_for_label:
        continue
    ax.annotate(
        f"{v}",
        xy=(ctr, b.get_height()), xytext=(0, label_offset_pts),
        textcoords="offset points",
        ha="center", va="bottom",
        rotation=val_rot_deg, rotation_mode="anchor",
        fontsize=font_sz, clip_on=False
    )

# Legend
handles = [mpatches.Patch(color=cmap(i), label=bin_labels[i]) for i in range(n_bins)]
leg = ax.legend(handles=handles, fontsize=font_sz, loc="upper right", frameon=False, title="Bin")
if leg and leg.get_title():
    leg.get_title().set_fontsize(font_sz)

# Save
plt.tight_layout()
plt.savefig(f"{save_basename}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{save_basename}.pdf", dpi=300, bbox_inches="tight")
# plt.show()
