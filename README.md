# Lab 3: Galaxy Morphology Classification with CNNs

Build a CNN to classify galaxy morphology from images. Given 64x64 color images from the Galaxy Zoo survey, your model classifies galaxies into 4 types: smooth round, smooth cigar, edge-on disk, and spiral.

## Learning Objectives

- Design and implement a CNN architecture in Flax
- Implement a training step with cross-entropy loss
- Understand how architecture choices affect accuracy on image data
- Exploit data symmetries to improve generalization

## Getting Started

### 1. Accept the assignment

Click the GitHub Classroom link shared by the instructor. This creates your own private copy of this repository under your GitHub account.

### 2. Clone your repository

Open a terminal and run:

```bash
git clone https://github.com/bu-ds595/lab-03-galaxy-cnn-YOUR_USERNAME.git
```

Replace `YOUR_USERNAME` with your actual GitHub username.

Then navigate into the folder:

```bash
cd lab-03-galaxy-cnn-YOUR_USERNAME
```

### 3. Install dependencies

From inside the lab folder, run:

```bash
pip install -r requirements.txt
```

If you get permission errors, try `pip install --user -r requirements.txt`.

### 4. Open the notebook

**Option A: VS Code**
1. Open VS Code
2. File -> Open Folder -> select the lab folder
3. Open `lab-03-galaxy-cnn.ipynb`
4. If prompted, install the Python and Jupyter extensions

**Option B: JupyterLab**
```bash
jupyter lab
```
Then click on `lab-03-galaxy-cnn.ipynb` in the file browser.

**Option C: Google Colab**

Upload the notebook, `cnn.py`, and `galaxy_data.npz` to [Google Colab](https://colab.research.google.com/). Add a cell at the top:
```python
! pip install jax jaxlib flax optax
```

## Exercises

Complete the `TODO` sections in `cnn.py`:

1. **`CNN` class** — Design your own CNN. Must accept `(batch, 64, 64, 3)` and return `(batch, 4)` logits.
2. **`train_step`** — Single gradient descent step with cross-entropy loss.
3. **Save your model** — After training, call `save_model(params)` to save `model_params.pkl`. The autograder loads this file to evaluate your model.

The notebook includes a self-check cell that replicates the autograder

## Running Tests Locally

```bash
pytest test_cnn.py -v
```

## Submitting Your Work

Save your notebook, `cnn.py`, and trained model, then commit and push:

```bash
git add lab-03-galaxy-cnn.ipynb cnn.py model_params.pkl
git commit -m "Complete lab 3"
git push
```

If `git push` asks for credentials, enter your GitHub username and a [personal access token](https://github.com/settings/tokens) (not your password).

You can push multiple times — only the final version at the deadline will be graded.

## Grading (4 points)

- **1 pt:** `train_step` correctly reduces loss
- **1 pt:** Saved model achieves >70% test accuracy
- **2 pts:** Saved model achieves >80% test accuracy — think about what symmetries the data has

## Data

[GalaxyMNIST](https://github.com/mwalmsley/galaxy_mnist) (Walmsley et al.), from Galaxy Zoo DECaLS Campaign A.
