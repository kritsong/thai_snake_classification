#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate experiment histories and generate comparison plots.

- CNN folders use legacy Keras history.json (require 'weightsTrue' in folder name).
- ViT/DeiT/Swin folders use HF Trainer logs (no 'weightsTrue' requirement).
- history.json formats supported:
    * Keras/legacy: {"val_accuracy": [...]}            -> series := val_accuracy
    * HF Trainer   : [ { "eval_accuracy": x, ... }, ]  -> series := eval_accuracy ordered by epoch

Training strategy labels are normalized to:
    - 'random initialization'  (previously 'full_train' / "from scratch")
    - 'linear probing'         (previously 'frozen' / "fixed feature extractor")
    - 'fine-tune'              (previously 'fineT' / 'correct_fine_tune')

Outputs figures to both:
    - test_final_comparison_figures/
    - final_combined_figures/

CHANGELOG (combined lines):
- Strategy, augmentation, and threshold line plots are COMBINED (CNN + Transformer on same axes).
- CNN curves use dashed ("--"), transformers use solid ("-"), and share colors per category.
- Robust color resolution for threshold categories (int vs str keys).
- Silence seaborn FutureWarning on first bar chart.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import inspect  # seaborn API compatibility
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# --- Global typography & axis thickness ---
plt.rcParams.update({
    "font.size": 14,
    "font.weight": "semibold",
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "axes.labelweight": "semibold",
    "axes.linewidth": 1.8,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,
    "grid.linewidth": 0.9,
    "legend.fontsize": 12,
})

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
EXPERIMENT_DIRS = [
    'cnn_experiment_results_ram_based',
    'transformer_experiment_results_deit_exact_aug',
    'transformer_experiment_results_swin_exact_aug',
    'transformer_experiment_results_vit',
]

# Save to BOTH old and new folders (as requested)
OUTPUT_DIRS = ['test_final_comparison_figures', 'final_combined_figures']
for d in OUTPUT_DIRS:
    os.makedirs(d, exist_ok=True)

HF_DIR_TOKENS = {
    'transformer_experiment_results_vit',
    'transformer_experiment_results_deit_exact_aug',
    'transformer_experiment_results_swin_exact_aug',
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def save_fig(filename: str, dpi: int = 300):
    """Save current Matplotlib figure to all output directories."""
    for d in OUTPUT_DIRS:
        path = os.path.join(d, filename)
        plt.savefig(path, dpi=dpi)

def stylize_axes(ax):
    """Apply thicker spines/ticks and tidy legend/frame."""
    for s in ax.spines.values():
        s.set_linewidth(1.8)
    ax.tick_params(width=1.6, length=6)
    leg = ax.get_legend()
    if leg is not None:
        leg.get_frame().set_linewidth(1.4)
        if leg.get_title() is not None:
            leg.get_title().set_fontweight('semibold')

def parse_experiment_name(name: str) -> dict:
    """Parse experiment folder name to extract model, family, strategy, augmentation, threshold."""
    params = {}

    # Model family / name (robust to slight naming variations)
    if ('ViT-Base' in name) or ('ViTBase16' in name) or name.startswith('ViT-'):
        params['model_family'] = 'Transformer'
        params['model_name'] = 'ViT'
    elif ('DeiT-Base' in name) or ('DeiT' in name):
        params['model_family'] = 'Transformer'
        params['model_name'] = 'DeiT'
    elif ('Swin-Base' in name) or ('SwinBase' in name) or ('Swin-' in name):
        params['model_family'] = 'Transformer'
        params['model_name'] = 'Swin'
    elif any(cnn in name for cnn in ['MobileNet', 'ResNet', 'EfficientNet']):
        params['model_family'] = 'CNN'
        params['model_name'] = name.split('_')[0]
    else:
        params['model_family'] = 'Unknown'
        params['model_name'] = 'Unknown'

    # Training strategy tokens (normalize to a small set; mapped to display later)
    n = name.lower()
    if 'full_train' in n or 'fromscratch' in n or 'scratch' in n:
        params['strategy'] = 'full_train'
    elif ('finet' in n) or ('fine-t' in n) or ('correct_fine_tune' in n) or ('correct-fine-tune' in n):
        params['strategy'] = 'fineT'
    elif 'frozen' in n or 'linearprobe' in n or 'linear-probe' in n:
        params['strategy'] = 'frozen'
    else:
        params['strategy'] = 'unknown'

    # Augmentation level (raw tokens; display relabeled in legends)
    if 'augnone' in n:
        params['aug'] = 'none'
    elif 'auglow' in n:
        params['aug'] = 'low'
    elif 'augmedium' in n:
        params['aug'] = 'medium'
    elif 'aughigh' in n:
        params['aug'] = 'high'
    else:
        params['aug'] = 'unknown'

    # Threshold (thres200 -> 200; default 0)
    try:
        params['threshold'] = int(next((p.replace('thres', '') for p in name.split('_') if p.startswith('thres')), '0'))
    except Exception:
        params['threshold'] = 0

    # Weights (may be absent in HF dirs)
    params['weights_flag'] = next((p.replace('weights', '') for p in name.split('_') if p.startswith('weights')), 'unknown')

    return params

def ensure_percent_scale(df: pd.DataFrame) -> pd.DataFrame:
    """If accuracies look like 0–1, convert to 0–100 (%). Also scales each 'history' list."""
    if df['best_acc'].max() <= 1.0 + 1e-12:
        df = df.copy()
        df['best_acc'] = df['best_acc'] * 100.0
        df['history'] = df['history'].apply(lambda h: [v * 100.0 for v in h])
    return df

def load_history(path: str):
    """Safe JSON loader."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def extract_accuracy_series(history_obj):
    """
    Return a list of validation accuracies for:
      * Keras-style dict: {'val_accuracy': [...]}        -> list
      * HF-style list of dicts: [{'eval_accuracy': x},]  -> list ordered by epoch
    """
    # Keras/legacy
    if isinstance(history_obj, dict) and 'val_accuracy' in history_obj:
        vals = history_obj['val_accuracy']
        if isinstance(vals, list) and vals:
            return [float(v) for v in vals]
        return None

    # HF/Trainer: list of dicts with eval points
    if isinstance(history_obj, list) and history_obj and isinstance(history_obj[0], dict):
        eval_points = []
        for rec in history_obj:
            if 'eval_accuracy' in rec:
                epoch = rec.get('epoch', np.nan)
                try:
                    epoch = float(epoch)
                except Exception:
                    epoch = np.nan
                acc = float(rec['eval_accuracy'])
                eval_points.append((epoch, acc))
        if not eval_points:
            return None
        eval_points.sort(key=lambda t: (float('inf') if np.isnan(t[0]) else t[0]))
        return [acc for _, acc in eval_points]

    return None

# --- Seaborn error-bar compatibility (SD lines, no caps) ----------------------
def bar_err_kw():
    """
    Return kwargs for seaborn.barplot to show SD error bars on all supported versions,
    with no caps (tips) and thicker lines.
    - seaborn >= 0.12: use errorbar='sd', err_kws={'linewidth': 1.6}
    - seaborn <  0.12: use ci='sd', errwidth=1.6
    """
    sig = inspect.signature(sns.barplot).parameters
    if 'errorbar' in sig:
        return dict(errorbar='sd', n_boot=None, capsize=0, err_kws={'linewidth': 1.6})
    else:
        return dict(ci='sd', errwidth=1.6, capsize=0)

ERR_KW = bar_err_kw()

# --- Robust color resolver (int/str keys) ------------------------------------
def _resolve_color_key(base, base_colors):
    """Return a color from base_colors for 'base' whether it's str or int."""
    if base in base_colors:
        return base_colors[base]
    try:
        as_int = int(base)
        if as_int in base_colors:
            return base_colors[as_int]
    except Exception:
        pass
    as_str = str(base)
    if as_str in base_colors:
        return base_colors[as_str]
    return 'gray'

# --- Generic history plotter (averages per category) -------------------------
def plot_average_history(
    dataframe: pd.DataFrame,
    study_variable: str,
    group_title: str,
    colors: dict,
    filename: str,
    sort_order=None,
    with_tail_inset: bool = False,
    tail_frac: float = 0.25,
    tail_min_points: int = 8,
    inset_size: tuple = (0.45, 0.45),  # width, height as fraction of axes
    inset_loc: int = 4,                # 1=UR, 2=UL, 3=LL, 4=LR
    ypad: float = 0.5,
    line_style_map: dict | None = None,     # per-category linestyle (optional)
    default_line_style: str = '-',          # fallback linestyle
    legend_title: str | None = None,        # legend title
    label_map: dict | None = None,          # map raw category -> display label
    legend_loc: str | int | None = None     # allow explicit legend placement
) -> None:
    """
    Plot average accuracy curves (MEAN ONLY; NO SD). If with_tail_inset=True,
    attach a zoomed inset focusing on the final part of the curves.
    """
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    categories = sorted(dataframe[study_variable].dropna().unique().tolist())
    if sort_order:
        categories = [c for c in sort_order if c in categories]

    # Compute averaged curves
    curves = {}
    max_len_overall = 0
    for category in categories:
        group = dataframe[dataframe[study_variable] == category]
        if group.empty:
            continue
        max_len = max(len(h) for h in group['history'])
        max_len_overall = max(max_len_overall, max_len)
        padded = [np.pad(np.asarray(h, dtype=float),
                         (0, max_len - len(h)),
                         mode='constant',
                         constant_values=np.nan)
                  for h in group['history']]
        stack = np.vstack(padded)
        avg_line = np.nanmean(stack, axis=0)
        x = np.arange(len(avg_line), dtype=float)
        curves[category] = (x, avg_line)

    # Main plot (thick lines)
    for category in categories:
        if category not in curves:
            continue
        x, avg_line = curves[category]
        color = _resolve_color_key(category, colors)
        ls = (line_style_map or {}).get(category, default_line_style)
        display_label = (label_map or {}).get(category, category)
        ax.plot(x, avg_line, linewidth=4, label=f"{display_label}", color=color, linestyle=ls)

    ax.set_title(
        f'{group_title}: Average Validation Accuracy over Epochs by {study_variable.replace("_", " ").title()}',
        fontsize=18, fontweight='bold'
    )
    ax.set_xlabel("Epoch (evaluation points)", fontsize=14, fontweight='semibold')
    ax.set_ylabel("Average Validation Accuracy (%)", fontsize=14, fontweight='semibold')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(
            handles, labels,
            title=legend_title or study_variable.replace('_', ' ').title(),
            loc=(legend_loc if legend_loc else 'best'),
            frameon=True
        )
    stylize_axes(ax)
    plt.tight_layout()

    # Optional tail inset (used only for augmentation plots)
    if with_tail_inset and max_len_overall > 0 and len(curves) > 0:
        tail_len = max(int(max_len_overall * tail_frac), tail_min_points)
        tail_len = min(tail_len, max_len_overall)
        tail_start = max_len_overall - tail_len
        tail_end = max_len_overall

        axins = inset_axes(ax, width=f"{int(inset_size[0]*100)}%", height=f"{int(inset_size[1]*100)}%",
                           loc=inset_loc, borderpad=1.0)

        ymins, ymaxs = [], []
        for category in categories:
            if category not in curves:
                continue
            x, avg_line = curves[category]
            tail_y = np.asarray(avg_line[tail_start:tail_end], dtype=float)
            valid = np.isfinite(tail_y)
            if not valid.any():
                continue
            ymins.append(np.nanmin(tail_y[valid]))
            ymaxs.append(np.nanmax(tail_y[valid]))
            color = _resolve_color_key(category, colors)
            ls = (line_style_map or {}).get(category, default_line_style)
            axins.plot(x[tail_start:tail_end], tail_y, linewidth=4.0, color=color, linestyle=ls)

        if ymins and ymaxs:
            ymin = min(ymins) - ypad
            ymax = max(ymaxs) + ypad
            axins.set_ylim(ymin, ymax)

        axins.set_xlim(tail_start, tail_end - 1)
        axins.grid(True, linestyle='--', alpha=0.4)
        for spine in axins.spines.values():
            spine.set_linewidth(1.6)
        axins.tick_params(labelsize=8, width=1.4, length=4)
        axins.set_title("Tail (zoom)", fontsize=10, pad=4, fontweight='semibold')
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=1.6)

    save_fig(filename)
    plt.show()

def plot_combined_by_family(
    df: pd.DataFrame,
    study_var: str,
    base_colors: dict,
    filename: str,
    legend_title: str,
    sort_order=None,
    with_tail_inset: bool = False,
    legend_loc: str | int | None = None,
    group_title: str = "All Models"
) -> None:
    """
    Combined plot for a given study variable (strategy/aug/threshold)
    overlaying CNN (dashed) and Transformer (solid) curves for each category,
    while sharing colors per category across families.
    """
    parts = []
    for fam in ['CNN', 'Transformer']:
        sub = df[df['model_family'] == fam].copy()
        if sub.empty:
            continue
        suffix = ' [CNN]' if fam == 'CNN' else ' [Transformer]'
        sub['__combo__'] = sub[study_var].astype(str) + suffix
        parts.append(sub)
    if not parts:
        return
    merged = pd.concat(parts, ignore_index=True)

    # Compute order expanded by family (Transformer first, then CNN)
    if sort_order:
        combo_order = []
        for cat in sort_order:
            combo_order.append(f"{cat} [Transformer]")
            combo_order.append(f"{cat} [CNN]")
        sort_order_use = [c for c in combo_order if c in merged['__combo__'].unique()]
    else:
        sort_order_use = None

    # Colors, linestyles, and label map
    categories = merged['__combo__'].unique().tolist()
    colors = {}
    line_styles = {}
    label_map = {}
    for cat in categories:
        base = cat.split(' [')[0]  # e.g., "low", "fine-tune", or "300"
        colors[cat] = _resolve_color_key(base, base_colors)
        if cat.endswith('[CNN]'):
            line_styles[cat] = '--'     # dashed for CNN
            label_map[cat] = f"{base} (CNN)"
        else:
            line_styles[cat] = '-'      # solid for Transformers
            label_map[cat] = f"{base} (Transformer)"

    plot_average_history(
        merged, '__combo__', group_title,
        colors, filename,
        sort_order=sort_order_use,
        with_tail_inset=with_tail_inset,
        line_style_map=line_styles,
        default_line_style='-',
        legend_title=legend_title,
        label_map=label_map,
        legend_loc=legend_loc
    )

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("Aggregating results from all experiment directories...")
    all_experiments_data = []

    for directory in EXPERIMENT_DIRS:
        if not os.path.exists(directory):
            print(f"Warning: Directory not found, skipping: {directory}")
            continue

        print(f"--> Loading from: {directory}")
        dir_lower = directory.lower()
        is_hf_dir = any(token in dir_lower for token in HF_DIR_TOKENS)
        require_weights_true = not is_hf_dir

        for experiment_name in os.listdir(directory):
            if require_weights_true and ('weightsTrue' not in experiment_name):
                continue
            experiment_path = os.path.join(directory, experiment_name)
            if not os.path.isdir(experiment_path):
                continue
            history_path = os.path.join(experiment_path, 'history.json')
            if not os.path.exists(history_path):
                continue

            hist = load_history(history_path)
            if hist is None:
                continue

            acc_series = extract_accuracy_series(hist)
            if not acc_series:
                continue

            params = parse_experiment_name(experiment_name)
            params['history'] = acc_series
            params['best_acc'] = float(np.nanmax(acc_series))
            all_experiments_data.append(params)

    if not all_experiments_data:
        print("\nNo valid experiment data found. Exiting.")
        return

    df = pd.DataFrame(all_experiments_data)
    df = ensure_percent_scale(df)
    df = df[df['model_name'] != 'Unknown'].copy()

    # Relabel strategies for display (normalize to journal terms)
    strategy_display_map = {
        'full_train': 'random initialization',
        'frozen':     'linear probing',
        'fineT':      'fine-tune',
        'unknown':    'unknown',
    }
    df['strategy'] = df['strategy'].map(strategy_display_map).fillna('unknown')

    print(f"\nSuccessfully loaded and processed {len(df)} total experiments.")

    # Orders & splits
    cnn_model_order = ['MobileNetV3Small', 'EfficientNetB0', 'ResNet50']
    transformer_model_order = ['ViT', 'DeiT', 'Swin']
    combined_model_order = cnn_model_order + transformer_model_order

    # Visual theme & palettes
    sns.set_theme(style="whitegrid")
    combined_model_colors = {
        'MobileNetV3Small': '#E69F00',
        'EfficientNetB0':   '#F0E442',
        'ResNet50':         '#D55E00',  # CNNs
        'ViT':  '#0072B2',
        'DeiT': '#56B4E9',
        'Swin': '#009E73'               # Transformers
    }

    # Legend titles
    LEGEND_TITLES = {
        'model_name': "Model Architecture",
        'strategy': "Training Strategy",
        'aug': "Augmentation level",
        'threshold': "Minimum Images per Class (Threshold)"
    }

    # Non-abbreviated augmentation legend entries
    AUG_DISPLAY_MAP = {
        'none':   'No augmentation',
        'low':    'Low augmentation',
        'medium': 'Medium augmentation',
        'high':   'High augmentation',
        'unknown':'Unknown augmentation'
    }

    # Strategy/aug orders & colors
    strategy_order = ['random initialization', 'linear probing', 'fine-tune']
    strategy_colors = {name: color for name, color in zip(strategy_order, sns.color_palette("rocket", n_colors=len(strategy_order)))}
    aug_order = ['none', 'low', 'medium', 'high']
    aug_colors = {name: color for name, color in zip(aug_order, sns.color_palette("viridis", n_colors=4))}
    threshold_order = sorted(df['threshold'].unique().tolist())
    threshold_colors = {name: color for name, color in zip(threshold_order, sns.color_palette("plasma", n_colors=len(threshold_order)))}

    # Linestyle map ONLY for model overlay: CNN dashed, Transformers solid
    combined_linestyles = {
        'MobileNetV3Small': '--',
        'EfficientNetB0':   '--',
        'ResNet50':         '--',
        'ViT':  '-',
        'DeiT': '-',
        'Swin': '-',
    }

    # -------------------------------------------------------------------------
    # BEST ACCURACY BAR CHARTS
    # -------------------------------------------------------------------------
    print("\n--- Generating Best Accuracy Bar Charts ---")

    # 1) Overall Model Performance (silence FutureWarning by using hue=x, dodge=False, legend=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=df,
        x='model_name', y='best_acc',
        hue='model_name', dodge=False, legend=False,
        estimator=np.mean,
        palette=combined_model_colors,
        order=combined_model_order,
        **ERR_KW
    )
    ax = plt.gca()
    ax.set_title('Overall Best Validation Accuracy by Model', fontsize=18, fontweight='bold')
    ax.set_ylabel('Average Best Validation Accuracy (%)', fontsize=14, fontweight='semibold')
    ax.set_xlabel('Model Architecture', fontsize=14, fontweight='semibold')
    stylize_axes(ax)
    plt.tight_layout()
    save_fig('barchart_01_overall_performance.png')
    plt.show()

    # 2) By Training Strategy (legend = Model Architecture)
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=df, x='strategy', y='best_acc',
        hue='model_name',
        estimator=np.mean,
        palette=combined_model_colors,
        order=strategy_order,
        hue_order=combined_model_order,
        **ERR_KW
    )
    ax = plt.gca()
    ax.set_title('Best Validation Accuracy by Training Strategy and Model', fontsize=18, fontweight='bold')
    ax.set_ylabel('Average Best Validation Accuracy (%)', fontsize=14, fontweight='semibold')
    ax.set_xlabel('Training Strategy', fontsize=14, fontweight='semibold')
    stylize_axes(ax)
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title("Model Architecture")
    plt.tight_layout()
    save_fig('barchart_02_by_strategy.png')
    plt.show()

    # 3) By Augmentation Level (legend = Model Architecture)
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=df, x='aug', y='best_acc',
        hue='model_name',
        estimator=np.mean,
        palette=combined_model_colors,
        order=aug_order,
        hue_order=combined_model_order,
        **ERR_KW
    )
    ax = plt.gca()
    ax.set_title('Best Validation Accuracy by Augmentation Level and Model', fontsize=18, fontweight='bold')
    ax.set_ylabel('Average Best Validation Accuracy (%)', fontsize=14, fontweight='semibold')
    ax.set_xlabel('Augmentation level', fontsize=14, fontweight='semibold')
    stylize_axes(ax)
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title("Model Architecture")
    plt.tight_layout()
    save_fig('barchart_03_by_augmentation.png')
    plt.show()

    # 4) By Data Threshold (legend = Model Architecture)
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=df, x='threshold', y='best_acc',
        hue='model_name',
        estimator=np.mean,
        palette=combined_model_colors,
        order=threshold_order,
        hue_order=combined_model_order,
        **ERR_KW
    )
    ax = plt.gca()
    ax.set_title('Best Validation Accuracy by Minimum Images per Class (Threshold) and Model', fontsize=18, fontweight='bold')
    ax.set_ylabel('Best Validation Accuracy (%)', fontsize=14, fontweight='semibold')
    ax.set_xlabel('Minimum Images per Class (Threshold)', fontsize=14, fontweight='semibold')
    stylize_axes(ax)
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title("Model Architecture")
    plt.tight_layout()
    save_fig('barchart_04_by_threshold.png')
    plt.show()

    # -------------------------------------------------------------------------
    # AVERAGE HISTORY LINE GRAPHS — COMBINED (CNN dashed, Transformer solid)
    # -------------------------------------------------------------------------
    print("\n--- Generating Combined Average History Line Graphs ---")

    # Combined by strategy
    plot_combined_by_family(
        df, 'strategy', strategy_colors,
        filename='linegraph_combined_by_strategy.png',
        legend_title="Training Strategy",
        sort_order=strategy_order,
        with_tail_inset=False,
        legend_loc='best',
        group_title="All Models"
    )

    # Combined by augmentation (with tail inset)
    plot_combined_by_family(
        df, 'aug', aug_colors,
        filename='linegraph_combined_by_augmentation.png',
        legend_title="Augmentation level",
        sort_order=aug_order,
        with_tail_inset=True,
        legend_loc='lower left',
        group_title="All Models"
    )

    # Combined by threshold
    plot_combined_by_family(
        df, 'threshold', threshold_colors,
        filename='linegraph_combined_by_threshold.png',
        legend_title="Minimum Images per Class (Threshold)",
        sort_order=threshold_order,
        with_tail_inset=False,
        legend_loc='best',
        group_title="All Models"
    )

    # -------------------------------------------------------------------------
    # EXTRA: Combined overlay (CNN dashed, Transformers solid) by MODEL
    # -------------------------------------------------------------------------
    print(" Generating Combined (CNN + Transformer) Average History Plot by Model...")
    plt.figure(figsize=(12, 8))
    model_colors = {k: combined_model_colors[k] for k in combined_model_order}
    # Reuse generic plotter directly for model_name, with dashed CNN vs solid Transformer
    plot_average_history(
        df, 'model_name', 'All Models',
        model_colors, 'linegraph_allmodels_by_model.png',
        sort_order=combined_model_order, with_tail_inset=False,
        line_style_map={
            'MobileNetV3Small': '--',
            'EfficientNetB0':   '--',
            'ResNet50':         '--',
            'ViT':  '-',
            'DeiT': '-',
            'Swin': '-',
        },
        default_line_style='-',
        legend_title="Model Architecture"
    )

    print(f"\nAll plots have been saved to: {', '.join(OUTPUT_DIRS)}")

if __name__ == '__main__':
    main()
