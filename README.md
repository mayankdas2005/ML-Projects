# ML

Small self-contained ML projects, each in its own notebook under `Projects/`.

## Setup

Dependencies are pinned in `pyproject.toml`/`uv.lock` and managed with
[uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone <this-repo-url>
cd ML
uv sync
```

`uv sync` creates `.venv` and installs the exact dependency versions from `uv.lock`
(Python 3.12, pinned via `.python-version`) — no separate `pip install` step needed.

## Reproducing results

Each notebook is written to be run top-to-bottom with no manual steps beyond data setup
(below): open it in VS Code or Jupyter, select the `.venv` interpreter/kernel this repo
created, and **Run All**. Every notebook seeds its randomness (`random_state=42` on
sklearn splits, `torch.manual_seed(42)` before any model/dataloader is built), so a fresh
run reproduces the same numbers reported in each project's README.

To run headlessly from the command line instead of an editor:

```bash
uv run --with jupyter jupyter nbconvert --to notebook --execute --inplace "Projects/<project>/<notebook>.ipynb"
```

## Projects

| Project | Notebook | Notes |
|---|---|---|
| [SMS Spam / Smishing Classifier](Projects/SMS-Spam-Classifier/README.md) | `SMS_Spam_Classifier_Baseline.ipynb` (TF-IDF + LogisticRegression), `SMS_Spam_Classifier_NN.ipynb` (mean-pooled embedding), `SMS_Spam_Classifier_CNN.ipynb`, `SMS_Spam_Classifier_FrozenEmbeddings.ipynb` (frozen MiniLM + LogisticRegression) | Data included in this repo. The frozen-embeddings notebook needs internet once to download its pretrained model. |
| [Fashion-MNIST](Projects/Fashion-MNIST/README.md) | `Fashion-MNIST.ipynb` | Requires a manual data download first — see that project's README. |

## Data

`Data/SMS-Spam-Data/` is committed directly (small enough to version). `Data/Fashion-MNIST/`
is gitignored — its training CSV alone is 133MB, over GitHub's 100MB per-file limit — see
[Projects/Fashion-MNIST/README.md](Projects/Fashion-MNIST/README.md) for the download link
and where to place the files.

`Pytorch-Tutorial/` and `main.py` are local scratch/learning material, not part of any
project, and are gitignored.
