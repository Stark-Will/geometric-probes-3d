---
license: mit
task_categories:
- image-classification
- other
language:
- en
tags:
- 3d
- 3d-generation
- mesh-evaluation
- human-annotation
- benchmark
- evaluation
pretty_name: DB-3DME
size_categories:
- 1K<n<10K
---

# DB-3DME: From Dataset to Benchmark for Human-aligned Automatic 3D Mesh Evaluation

DB-3DME is a benchmark dataset for evaluating 3D mesh generation, featuring human annotations for Geometry and Prompt Adherence, along with corresponding text prompts and GIF visualizations of generated 3D assets. It is intended to facilitate research on reliable evaluation protocols for modern 3D generative models.

## Dataset Description

Each entry in the dataset links a text prompt to a generated 3D mesh (visualized as a rotating GIF) and provides human ratings on two axes:

- **Geometry Rating** — quality of the generated geometry
- **Prompt Adherence Rating** — how well the mesh aligns with the input prompt

## Dataset Structure

### Files

- `metadata.csv` — Metadata and human evaluation results for each 3D asset
- `GIFs/` — Directory containing GIF visualizations of the 3D assets

### Data Fields

| Column | Description |
|---|---|
| `Prompt` | Text prompt used to generate the 3D asset |
| `gif_path` | Path to the GIF visualization of the 3D mesh (relative to repo root) |
| `Geometry Rating` | Human rating of geometric quality |
| `Prompt Adherence Rating` | Human rating of prompt alignment |

### Example

![Example](GIFs/output-203.gif)

- **Prompt:** SUV
- **Geometry Rating:** 2
- **Prompt Adherence Rating:** 3

## Usage

### Load with the `datasets` library

```python
from datasets import load_dataset

ds = load_dataset("nsjia/DB-3DME")
samples = ds["train"][0]
```

### Load the metadata directly with `pandas`

```python
from huggingface_hub import hf_hub_download
import pandas as pd

csv_path = hf_hub_download(
    repo_id="nsjia/DB-3DME",
    filename="metadata.csv",
    repo_type="dataset",
)
df = pd.read_csv(csv_path)
```

### Download the full dataset

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="nsjia/DB-3DME",
    repo_type="dataset",
)
```

### Convert GIFs to Grid Images

A utility script `gif_to_grid.py` (available in the [companion GitHub repo](https://github.com/nsjia/DB-3DME)) converts a 3D asset GIF into a single grid image for convenient visualization and evaluation.


## Citation

If you use this dataset, please cite:

```bibtex
@article{db3dme,
  title={DB-3DME: From Dataset to Benchmark for Human-aligned Automatic 3D Mesh Evaluation},
  author={...},
  year={2026}
}
```

## License

Please refer to the license file in the repository.

## Links

- 📄 **Paper:** [link to paper]
- 💻 **Code:** https://github.com/nsjia/DB-3DME