import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib
import numpy as np
from sklearn.datasets import load_digits

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def ensure_project_directories(project_root: Path) -> None:
    """Create the folders used by the project if they do not exist."""
    (project_root / ".matplotlib").mkdir(exist_ok=True)
    (project_root / "plots").mkdir(exist_ok=True)
    (project_root / "data").mkdir(exist_ok=True)


def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer labels like 3 into one-hot vectors like [0,0,0,1,...]."""
    if labels.ndim != 1:
        raise ValueError("Labels must be a 1D array for one-hot encoding.")

    if num_classes <= 0:
        raise ValueError("num_classes must be greater than 0.")

    encoded_labels = np.zeros((labels.shape[0], num_classes))
    encoded_labels[np.arange(labels.shape[0]), labels] = 1
    return encoded_labels


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    images: np.ndarray,
    test_ratio: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle the dataset and split it into training and test parts."""
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1.")

    if len(X) != len(y) or len(X) != len(images):
        raise ValueError("X, y, and images must contain the same number of samples.")

    rng = np.random.default_rng(random_seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)

    test_size = int(len(X) * test_ratio)
    if test_size == 0 or test_size == len(X):
        raise ValueError("test_ratio creates an invalid split size.")

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    train_images = images[train_indices]
    test_images = images[test_indices]

    return X_train, X_test, y_train, y_test, train_images, test_images


def load_and_prepare_digits(
    test_ratio: float = 0.2,
    random_seed: int = 42,
) -> dict[str, np.ndarray]:
    """Load the sklearn digits dataset and prepare it for the NumPy model."""
    digits = load_digits()

    if digits.data is None or digits.target is None or digits.images is None:
        raise ValueError("The digits dataset could not be loaded correctly.")

    X = digits.data.astype(np.float64)
    y = digits.target.astype(int)
    images = digits.images.astype(np.float64)

    # Pixel values in this dataset go from 0 to 16, so divide by 16 to normalize.
    X = X / 16.0

    X_train, X_test, y_train_labels, y_test_labels, _, test_images = split_dataset(
        X=X,
        y=y,
        images=images,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )

    y_train = one_hot_encode(y_train_labels, num_classes=10)
    y_test = one_hot_encode(y_test_labels, num_classes=10)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_labels": y_train_labels,
        "y_test_labels": y_test_labels,
        "test_images": test_images,
    }


def calculate_accuracy(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> float:
    """Calculate classification accuracy."""
    if true_labels.shape != predicted_labels.shape:
        raise ValueError("true_labels and predicted_labels must have the same shape.")

    return float(np.mean(true_labels == predicted_labels))


def plot_training_history(history: dict[str, list[float]], plots_dir: Path) -> None:
    """Save the training loss and training accuracy graphs."""
    loss_history = history.get("loss", [])
    accuracy_history = history.get("accuracy", [])

    if not loss_history or not accuracy_history:
        raise ValueError("Training history is empty, so there is nothing to plot.")

    epochs = np.arange(1, len(loss_history) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss_history, color="tab:blue", linewidth=2)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "loss_curve.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, np.array(accuracy_history) * 100, color="tab:green", linewidth=2)
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "accuracy_curve.png")
    plt.close()


def save_sample_predictions(
    images: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    plots_dir: Path,
    num_samples: int = 6,
) -> None:
    """Save a small grid of sample predictions from the test set."""
    if num_samples <= 0:
        raise ValueError("num_samples must be greater than 0.")

    sample_count = min(num_samples, len(images))
    rows = 2
    columns = int(np.ceil(sample_count / rows))

    plt.figure(figsize=(10, 5))

    for index in range(sample_count):
        plt.subplot(rows, columns, index + 1)
        plt.imshow(images[index], cmap="gray")
        plt.title(f"Pred: {predicted_labels[index]}\nTrue: {true_labels[index]}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(plots_dir / "sample_predictions.png")
    plt.close()
