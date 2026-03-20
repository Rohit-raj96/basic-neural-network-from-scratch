from pathlib import Path

import numpy as np

from model import NeuralNetwork
from utils import (
    calculate_accuracy,
    ensure_project_directories,
    load_and_prepare_digits,
    plot_training_history,
    save_sample_predictions,
)


# Beginner-friendly configuration values
INPUT_SIZE = 64
HIDDEN_SIZE = 32
OUTPUT_SIZE = 10
LEARNING_RATE = 0.1
EPOCHS = 200
HIDDEN_ACTIVATION = "relu"
TEST_RATIO = 0.2
RANDOM_SEED = 42
NUM_SAMPLE_PREDICTIONS = 6


def main() -> None:
    project_root = Path(__file__).resolve().parent
    plots_dir = project_root / "plots"

    ensure_project_directories(project_root)

    try:
        dataset = load_and_prepare_digits(
            test_ratio=TEST_RATIO,
            random_seed=RANDOM_SEED,
        )
    except Exception as error:
        print(f"Dataset loading failed: {error}")
        return

    X_train = dataset["X_train"]
    X_test = dataset["X_test"]
    y_train = dataset["y_train"]
    y_test = dataset["y_test"]
    y_train_labels = dataset["y_train_labels"]
    y_test_labels = dataset["y_test_labels"]
    test_images = dataset["test_images"]

    print("Dataset loaded successfully.")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Input shape: {X_train.shape}")
    print(f"Output shape: {y_train.shape}")
    print("-" * 60)

    model = NeuralNetwork(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        learning_rate=LEARNING_RATE,
        activation=HIDDEN_ACTIVATION,
        random_seed=RANDOM_SEED,
    )

    history = model.train(
        X_train=X_train,
        y_train=y_train,
        epochs=EPOCHS,
        print_every=1,
    )

    test_probabilities = model.predict_proba(X_test)
    test_predictions = np.argmax(test_probabilities, axis=1)
    test_loss = model.compute_loss(y_test, test_probabilities)
    test_accuracy = calculate_accuracy(y_test_labels, test_predictions)

    print("-" * 60)
    print("Test set evaluation")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    try:
        plot_training_history(history, plots_dir)
        save_sample_predictions(
            images=test_images,
            true_labels=y_test_labels,
            predicted_labels=test_predictions,
            plots_dir=plots_dir,
            num_samples=NUM_SAMPLE_PREDICTIONS,
        )
    except Exception as error:
        print(f"Plot saving failed: {error}")

    print("-" * 60)
    print("Sample predictions")
    for index in range(min(NUM_SAMPLE_PREDICTIONS, len(test_predictions))):
        print(
            f"Sample {index + 1}: predicted = {test_predictions[index]}, "
            f"actual = {y_test_labels[index]}"
        )

    print("-" * 60)
    print("Saved files")
    print(f"Loss graph: {plots_dir / 'loss_curve.png'}")
    print(f"Accuracy graph: {plots_dir / 'accuracy_curve.png'}")
    print(f"Sample predictions graph: {plots_dir / 'sample_predictions.png'}")


if __name__ == "__main__":
    main()
