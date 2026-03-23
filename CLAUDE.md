# EcoPack Project

## Overview
EcoPack is a GPU co-scheduling system that packs multi-GPU jobs to reduce idle GPU time and energy waste. The key insight is that more GPUs don't always mean better performance — some apps scale poorly, so EcoPack assigns fewer GPUs to those apps and co-schedules multiple jobs simultaneously.

## Project Structure
- `data/` — per-system (V100, A100, H100) performance metrics (`perf_metrics.txt`) with runtime, power, DRAM, SM, FP counters across 1–4 GPU configs
- `plot/` — Jupyter notebooks for paper figures
- `fig/` — output directory for saved figures (PNG, 300 DPI)
- `exp/` — experiment scripts
- `results/` — experiment results

## Key Data Files
- `data/{V100,A100,H100}/perf_metrics.txt` — columns: `gpu_count runtime_s avg_power dram_sum sm_sum fp_sum`, grouped by app sections (`===== category/appname =====`)

## Important Rules

### When editing notebooks
- **NEVER overwrite user's manual code changes.** Always read the current cell state first, then only add/modify what was explicitly requested.
- The user frequently tweaks font sizes, figure sizes, spine settings, and other styling by hand. Preserve all of these.
- Use `NotebookEdit` with the correct `cell_id` — always read the notebook first to confirm the current cell IDs and content.

### Plotting conventions
- `plt.rcParams`: `font.family: serif`
- Figures saved to `../fig/` or `fig/` directory at 300 DPI with `bbox_inches='tight'`
- The user prefers clean, publication-quality figures
- Use `ax.tick_params(axis='x', labelsize=N)` instead of passing `fontsize` to `set_xticks()` (compatibility with older matplotlib)

### Before making changes
- Read the current file/cell state before editing
- Don't assume previous edits are still in place — the user may have changed things manually
- When the user says "keep my code" or similar, be extra careful to preserve everything unchanged
- Don't assume it, if you are unsure, ask me before any changes or come up with ideas. 
