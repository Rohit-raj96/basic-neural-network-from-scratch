# Basic Neural Network from Scratch

This is a beginner-friendly Week 1 internship project for digit classification using a neural network built from scratch in Python.

The neural network logic is implemented with NumPy only. The `sklearn` digits dataset is used only for loading the dataset. No ML frameworks such as TensorFlow, PyTorch, Keras, or sklearn's neural-network models are used.

## Project Features

- Feedforward neural network with one hidden layer
- Forward propagation
- ReLU or sigmoid activation for the hidden layer
- Softmax activation for the output layer
- Cross-entropy loss
- Manual backpropagation using NumPy
- Gradient descent weight updates
- Training loss and accuracy tracking
- Test set evaluation
- Sample prediction visualization

## Project Structure

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

## Dataset

This project uses the built-in `digits` dataset from `sklearn.datasets`.

- Total samples: 1797
- Image size: `8 x 8`
- Flattened input size: `64`
- Number of classes: `10` digits (`0` to `9`)

No separate download is needed. Installing `scikit-learn` is enough.

## Input Shape and Output Shape

- Each digit image is `8 x 8`
- The image is flattened into a vector of length `64`
- So the input shape for training data is `(number_of_samples, 64)`
- The labels are one-hot encoded into 10 values
- So the output shape is `(number_of_samples, 10)`

Example:

- Input sample shape: `(64,)`
- Batch input shape: `(1438, 64)` approximately
- Output sample shape: `(10,)`

## Model Architecture

```text
Input Layer:   64 neurons
Hidden Layer:  32 neurons
Output Layer:  10 neurons
```

Hidden layer activation:

- `relu` by default
- `sigmoid` is also implemented and can be used

Output layer activation:

- `softmax`

## How Weights and Biases Are Initialized

- `W1` and `W2` are initialized with small random values
- `b1` and `b2` are initialized with zeros
- For ReLU, the first layer uses He-style scaling: `sqrt(2 / input_size)`
- For sigmoid, the first layer uses Xavier-style scaling: `sqrt(1 / input_size)`

This helps training start in a stable way instead of using weights that are too large or too small.

## Why Backpropagation Is Needed

Backpropagation is the process that tells the network how much each weight and bias contributed to the error.

Without backpropagation:

- the model can make predictions
- but it cannot learn from mistakes

With backpropagation:

- we compute gradients of the loss with respect to weights and biases
- we move parameters in the opposite direction of the gradients
- the loss decreases over time
- the model becomes better at classification

In short, backpropagation is what makes training possible.

## How Training Works

1. Load the digits dataset
2. Normalize input values by dividing by `16.0`
3. One-hot encode the labels
4. Run forward propagation
5. Compute cross-entropy loss
6. Run backpropagation
7. Update weights and biases using gradient descent
8. Repeat for many epochs
9. Plot loss and accuracy
10. Evaluate on the test set

## VS Code Setup and Run Instructions

Open the project folder in VS Code, then open the integrated terminal and run these commands.

### Windows PowerShell

```powershell
cd basic-neural-network-from-scratch
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### If you already have the existing virtual environment in this folder

```powershell
cd basic-neural-network-from-scratch
.\venv\Scripts\Activate.ps1
python main.py
```

## What the Program Does

When you run `main.py`, it will:

- load and preprocess the dataset
- train the neural network
- print epoch-wise loss and accuracy
- evaluate on the test data
- save graphs inside the `plots/` folder
- save a sample predictions image inside the `plots/` folder

## Output Files

After training, these files are saved:

- `plots/loss_curve.png`
- `plots/accuracy_curve.png`
- `plots/sample_predictions.png`

## Changing the Hidden Activation

In `main.py`, change this line:

```python
HIDDEN_ACTIVATION = "relu"
```

You can switch it to:

```python
HIDDEN_ACTIVATION = "sigmoid"
```

## Common Mistakes and Fixes

### 1. `ModuleNotFoundError`

Cause:

- required packages are not installed

Fix:

```powershell
pip install -r requirements.txt
```

### 2. PowerShell script execution is disabled

Cause:

- PowerShell blocks virtual environment activation

Fix:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 3. Graphs do not appear in the folder

Cause:

- training stopped early due to an error
- the `plots/` folder was not created yet

Fix:

- run `python main.py` again
- check the error shown in the terminal

### 4. Wrong Python interpreter in VS Code

Cause:

- VS Code may be using global Python instead of the project virtual environment

Fix:

- press `Ctrl + Shift + P`
- choose `Python: Select Interpreter`
- select the interpreter inside the project's `venv`

## Notes

- This project uses full-batch gradient descent for simplicity
- The goal is clarity and learning, not production-level optimization
- The neural network implementation is intentionally kept small and modular
