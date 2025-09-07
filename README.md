# Detecting-Food-Adulteration-from-Analytical-Data-via-a-Graph-based-Semi-supervised-Approach

This repository implements a reproducible three-stage pipeline:

1. **Feature selection (LASSO)**
2. **Key-sample marking** using MST + spectral clustering (under three distance metrics)
3. **Semi-supervised learning** with a KNN graph + **Measure Propagation** (MP), plus **CCER** (Cross-Class Edge Ratio) to assess cluster separation

Designed to run across multiple datasets (e.g., *Groundnut\_oil*, *Milk*, *Strawberry\_purees*) with minimal path changes.

---

## Repository Layout

```
.
├─ code/
│  ├─ feature_selection_lasso.py
│  ├─ key_sample_three_sim.py
│  ├─ key_sample_three_sim_filtered.py
│  ├─ semi_three_sim.py
│  └─ semi_three_sim_filtered.py
├─ Dataset/
│  ├─ Groundnut_oil.csv
│  ├─ Milk.csv
│  └─ Strawberry_purees.csv
└─ outputs/                     # created at runtime for CSV/XLSX/PNG artifacts
```

> **Case matters**: if scripts refer to `Dataset/`, keep the folder name exactly `Dataset/` (not `dataset/`).

---

## Data Format

Input CSVs must contain:

* `Wavelength` — an ID/descriptor column (any readable identifier)
* Numeric feature columns
* `Class` — integer class label. After key-sample marking, **non-key** samples will be set to `-1` (unlabeled) for the semi-supervised stage.

---

## Scripts Overview

### `code/feature_selection_lasso.py`

* Reads a CSV from `Dataset/...`, standardizes features, runs **LASSO** (`alpha` tunable), keeps **non-zero** coefficients, and writes a filtered CSV to `outputs/.../Groundnut_mst_filtered.csv`.

### `code/key_sample_three_sim.py` (raw data)

* Computes three distance matrices: **Euclidean / Manhattan / Cosine**.
* For each distance: builds an **MST**, converts to an adjacency matrix, runs **spectral clustering** (`n_clusters = 2`), and selects **key samples** per community.
* **Default selection rule**: nodes with **degree ≥ 95th percentile** are kept as labeled; others are set to `Class = -1`.
* Writes three XLSX files: `..._Euclidean.xlsx`, `..._Manhattan.xlsx`, `..._Cosine.xlsx`.

### `code/key_sample_three_sim_filtered.py` (filtered data)

* Same as above, but **starts from the LASSO-filtered CSV**.

### `code/semi_three_sim.py` (raw data)

* Reads the three XLSX files, standardizes features, selects the **similarity function by filename** (Euclidean/Manhattan/Cosine), builds a **KNN** graph, and runs **Measure Propagation**.
* Saves per-iteration `results_*.csv` (neighbors, original/pred labels, **CCER**) and per-K `metrics_*.csv` (accuracy/recall/precision/F1/AUC/**CCER**), plus a labeled graph PNG.

### `code/semi_three_sim_filtered.py` (filtered data)

* Same as `semi_three_sim.py`, but operates on the **filtered** XLSX outputs and writes to a filtered results directory under `outputs/`.

---

## Choosing the **Proportion of Key Samples** (10% / 20% / 50%)

**Default** selection in the key-sample scripts:

> Within each community, keep nodes with **degree ≥ 95th percentile** as labeled; set others to `Class = -1`.

To **control the labeled ratio directly**, switch to a **top-k%** rule. In `key_sample_three_sim*.py`, replace the percentile line with:

```python
# Default (percentile-based):
# filtered_nodes = [node for node in sorted_nodes if degrees[node] >= degree_threshold]

# Fixed-ratio alternative (example: 20%)
KEY_RATIO = 0.20                  # change to 0.10 / 0.20 / 0.50 as needed
top_k_count = max(1, int(len(sorted_nodes) * KEY_RATIO))
filtered_nodes = sorted_nodes[:top_k_count]
```

This selects the top-`KEY_RATIO` fraction of high-degree nodes **per community** as known labels.

---

## CCER: Cross-Class Edge Ratio

**Definition**

$$
\mathrm{CCER} \;=\; \frac{\sum_{(v_i, v_j)\in E}\mathbf{1}\{\hat{y}_i \neq \hat{y}_j\}}{|E|}
$$

where $E$ are KNN edges (the implementation counts **directed** edges $i \to j$).

* **CCER = 0**: no edge connects different predicted classes → perfectly separated clusters; the result can be treated as final.
* **CCER > 0**: cross-class edges remain → boundaries are fuzzy.

**Stored in outputs**

* Each `results_*.csv` includes a `CCER` column (same scalar for that iteration).
* Each `metrics_*.csv` includes a `ccer` column alongside accuracy/recall/precision/F1/AUC.

---

## Hyperparameters

### 1) LASSO (`feature_selection_lasso.py`)

* `alpha = 0.01` by default. Increase for stronger sparsity, or grid-search (e.g., `[0.001, 0.01, 0.1]`).
* Features are standardized with `StandardScaler()`.

### 2) Key-sample marking (`key_sample_three_sim*.py`)

* `n_clusters = 2` for spectral clustering (increase for multi-class datasets).
* Selection strategy:

  * **Default**: degree ≥ **95th percentile**
  * **Optional**: fixed ratio `KEY_RATIO` = 0.10 / 0.20 / 0.50 (see snippet above)
* Visualization uses `spring_layout(seed=0)` for reproducible figures.

### 3) Semi-supervised (`semi_three_sim*.py`)

* **KNN `K`**: `K_values = [3, 4, 5]` by default. Denser graphs may increase cross-class edges—use **CCER** to choose `K`.
* **Similarity**: chosen by filename token (`Euclidean` / `Manhattan` / `Cosine`). Cosine uses `cosine_distances` and maps to (0, 1].
* **Measure Propagation grid**:

  * `mu`: neighbor message strength (e.g., `[1e-8, 1e-4, 0.01, 0.1, 1, 10, 100]`)
  * `nu`: local regularization (e.g., `[1e-8, 1e-6, 1e-4, 0.01, 0.1]`)
  * `max_iter`: default `1000`
  * `tol`: default `1e-1` (early stop when relative change is small)
* **Metrics**: accuracy / macro recall / macro precision / macro F1 / AUC (`multi_class='ovo'`) + **CCER**.

---

## Quick Start

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Example `requirements.txt`:

   ```
   pandas
   numpy
   scikit-learn
   scipy
   matplotlib
   networkx
   adjustText
   tqdm
   ```

2. **Place your data** under `Dataset/` (e.g., `Groundnut_oil.csv`, `Milk.csv`, `Strawberry_purees.csv`).
   Ensure columns include `Wavelength`, numeric features, and `Class`.

3. **Run one of the two pipelines**

**A. Raw-data route**

```bash
# Mark key samples under three metrics (produces 3 XLSX files)
python code/key_sample_three_sim.py

# Semi-supervised KNN + MP + CCER on the three XLSX files
python code/semi_three_sim.py
```

**B. Filtered-data route (with LASSO)**

```bash
# LASSO feature selection -> outputs/.../Groundnut_mst_filtered.csv
python code/feature_selection_lasso.py

# Key-sample marking on the filtered CSV -> three XLSX files
python code/key_sample_three_sim_filtered.py

# Semi-supervised KNN + MP + CCER on the filtered XLSX files
python code/semi_three_sim_filtered.py
```

**Outputs** (under `outputs/`):

* `metrics_*.csv` — per-K aggregated metrics (including `ccer`)
* `results_*.csv` — per-iteration details (neighbors, original/pred labels, `CCER`)
* `*.png` — KNN graph visualizations with labels

---

## Tips & Troubleshooting

* **AUC errors/NaN** may occur if predictions contain only a single class. Inspect label balance or rely on other metrics/CCER for that run.
* **Case sensitivity**: `Dataset/` ≠ `dataset/`. Keep paths consistent with folder names.
* **Speed/memory**: Large `K` or a wide grid can be slow on large datasets. Start coarse, review **CCER** and metrics, then refine.
* **Multi-class**: Increase `n_clusters` in the key-sample step and keep `multi_class='ovo'` for AUC.
* **Reproducible figures**: graph layout uses a fixed seed.

---

## License & Citation (optional)

Add a `LICENSE` of your choice. If you use this repository in academic work, please cite the repo and your dataset sources. You may also acknowledge the **CCER** criterion and the graph-based semi-supervised framework implemented here.
