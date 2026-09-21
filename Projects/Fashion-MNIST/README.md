# Fashion-MNIST

A small feedforward network (`nn.Linear` -> ReLU -> `nn.Linear` -> ReLU -> `nn.Linear`)
classifying Fashion-MNIST images, trained with plain SGD for 100 epochs.

## Data setup

The CSVs aren't committed to this repo (`fashion-mnist_train.csv` alone is 133MB, over
GitHub's 100MB per-file limit). Download them before running the notebook:

1. Get `fashion-mnist_train.csv` and `fashion-mnist_test.csv` from Kaggle:
   https://www.kaggle.com/datasets/zalando-research/fashionmnist
2. Place both files in `Data/Fashion-MNIST/` at the repo root, so you end up with:
   - `Data/Fashion-MNIST/fashion-mnist_train.csv`
   - `Data/Fashion-MNIST/fashion-mnist_test.csv`

Then Run All in `Fashion-MNIST.ipynb`. The notebook seeds `torch.manual_seed(42)` before
building the model or dataloaders, so results are reproducible run to run.
