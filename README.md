# Basic Neural Network from Scratch

This project is a simple handwritten digit classifier that I built from scratch using Python and NumPy.

I made it as a learning project to understand how a neural network works internally instead of using ready-made deep learning libraries. The main focus of this project is to practice the full flow step by step: forward propagation, loss calculation, backpropagation, parameter updates, and evaluation.

## About the project

- Built with `Python` and `NumPy`
- Uses the `digits` dataset from `scikit-learn`
- Classifies digits from `0` to `9`
- Trains a neural network with one hidden layer
- Saves loss and accuracy graphs after training
- Saves a sample prediction image in the `plots/` folder

`scikit-learn` is used only for loading the dataset. The neural network itself is implemented manually.

## Project structure

```text
basic-neural-network-from-scratch/
|-- data/
|   |-- README.md
|-- plots/
|   |-- .gitkeep
|-- main.py
|-- model.py
|-- utils.py
|-- requirements.txt
|-- README.md
|-- .gitignore
```

## Dataset details

This project uses the built-in `digits` dataset from `sklearn.datasets`.

- Total samples: `1797`
- Image size: `8 x 8`
- Input size after flattening: `64`
- Output classes: `10`

No manual dataset download is needed.

## Model details

The model used in this project is a basic feedforward neural network:

```text
Input layer:   64 neurons
Hidden layer:  32 neurons
Output layer:  10 neurons
```

- Hidden activation: `relu`
- Optional hidden activation: `sigmoid`
- Output activation: `softmax`
- Loss function: `cross-entropy`
- Learning method: `gradient descent`

## How the project works

1. Load the digits dataset
2. Normalize the input values
3. Convert labels into one-hot encoded vectors
4. Run forward propagation
5. Calculate loss
6. Run backpropagation
7. Update weights and biases
8. Repeat for multiple epochs
9. Evaluate on test data
10. Save graphs and sample predictions

## Files used in the project

- `main.py` runs the full project
- `model.py` contains the neural network class
- `utils.py` contains helper functions for dataset loading, preprocessing, accuracy, and plotting
- `data/README.md` contains a short note about the dataset folder
- `plots/` stores the generated output images

## How to run

Open the project folder in VS Code or terminal and run:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

If PowerShell blocks activation, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## Output

After running the project, these files are created inside the `plots/` folder:

- `loss_curve.png`
- `accuracy_curve.png`
- `sample_predictions.png`

The terminal also shows:

- training loss for each epoch
- training accuracy for each epoch
- final test loss
- final test accuracy
- a few sample predictions

## Changing the activation function

In `main.py`, this line controls the hidden layer activation:

```python
HIDDEN_ACTIVATION = "relu"
```

You can change it to:

```python
HIDDEN_ACTIVATION = "sigmoid"
```

## What I learned from this project

- how forward propagation works
- how backpropagation updates weights
- how one-hot encoding is used in classification
- how loss and accuracy change during training
- how to organize a small machine learning project clearly

## Note

This project is made for learning and practice. The aim is to understand the logic of a neural network clearly, not to build a production-level model.
