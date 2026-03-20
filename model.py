import numpy as np


class NeuralNetwork:
    """A simple feedforward neural network with one hidden layer."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        learning_rate: float = 0.1,
        activation: str = "relu",
        random_seed: int = 42,
    ) -> None:
        if input_size <= 0 or hidden_size <= 0 or output_size <= 0:
            raise ValueError("Layer sizes must be positive integers.")

        if learning_rate <= 0:
            raise ValueError("Learning rate must be greater than 0.")

        if activation not in {"relu", "sigmoid"}:
            raise ValueError("Activation must be either 'relu' or 'sigmoid'.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation_name = activation
        self.random_generator = np.random.default_rng(random_seed)

        self.W1, self.b1, self.W2, self.b2 = self._initialize_parameters()

    def _initialize_parameters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Initialize weights with small random values and biases with zeros.

        The weights are initialized with a small random value using the
        standard normal distribution, while the biases are initialized with
        zeros.

        We use different scaling factors for the first and second layer
        depending on the activation function used. For ReLU, the first
        layer uses He-style scaling: sqrt(2 / input_size). For sigmoid,
        the first layer uses Xavier-style scaling: sqrt(1 / input_size).
        """
        if self.activation_name == "relu":
            # He-style scaling for ReLU
            first_layer_scale = np.sqrt(2.0 / self.input_size)
        else:
            # Xavier-style scaling for sigmoid
            first_layer_scale = np.sqrt(1.0 / self.input_size)

        # Xavier-style scaling for the second layer
        second_layer_scale = np.sqrt(1.0 / self.hidden_size)

        # Initialize weights with small random values
        W1 = self.random_generator.standard_normal(
            (self.input_size, self.hidden_size)
        ) * first_layer_scale
        # Initialize biases with zeros
        b1 = np.zeros((1, self.hidden_size))

        # Initialize weights with small random values
        W2 = self.random_generator.standard_normal(
            (self.hidden_size, self.output_size)
        ) * second_layer_scale
        # Initialize biases with zeros
        b2 = np.zeros((1, self.output_size))

        return W1, b1, W2, b2

    @staticmethod
    def relu(values: np.ndarray) -> np.ndarray:
        return np.maximum(0, values)

    @staticmethod
    def relu_derivative(values: np.ndarray) -> np.ndarray:
        return (values > 0).astype(float)

    @staticmethod
    def sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-values))

    @staticmethod
    def sigmoid_derivative(activated_values: np.ndarray) -> np.ndarray:
        return activated_values * (1.0 - activated_values)

    @staticmethod
    def softmax(values: np.ndarray) -> np.ndarray:
        shifted_values = values - np.max(values, axis=1, keepdims=True)
        exp_values = np.exp(shifted_values)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def _apply_hidden_activation(self, values: np.ndarray) -> np.ndarray:
        if self.activation_name == "relu":
            return self.relu(values)
        return self.sigmoid(values)

    def _hidden_activation_derivative(
        self,
        linear_values: np.ndarray,
        activated_values: np.ndarray,
    ) -> np.ndarray:
        if self.activation_name == "relu":
            return self.relu_derivative(linear_values)
        return self.sigmoid_derivative(activated_values)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Run forward propagation and store intermediate values for backpropagation."""
        if X.ndim != 2 or X.shape[1] != self.input_size:
            raise ValueError(
                f"Expected input shape (batch_size, {self.input_size}), "
                f"but received {X.shape}."
            )

        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._apply_hidden_activation(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.softmax(z2)

        cache = {
            "X": X,
            "z1": z1,
            "a1": a1,
            "z2": z2,
            "a2": a2,
        }
        return a2, cache

    @staticmethod
    def compute_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"y_true shape {y_true.shape} does not match y_pred shape {y_pred.shape}."
            )

        epsilon = 1e-12
        clipped_predictions = np.clip(y_pred, epsilon, 1.0)
        loss = -np.sum(y_true * np.log(clipped_predictions)) / y_true.shape[0]
        return float(loss)

    @staticmethod
    def _calculate_accuracy_from_one_hot(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        true_labels = np.argmax(y_true, axis=1)
        predicted_labels = np.argmax(y_pred, axis=1)
        return float(np.mean(true_labels == predicted_labels))

    def backward(
        self,
        y_true: np.ndarray,
        cache: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Run backpropagation and return gradients for all parameters."""
        X = cache["X"]
        z1 = cache["z1"]
        a1 = cache["a1"]
        a2 = cache["a2"]

        number_of_samples = X.shape[0]

        dZ2 = a2 - y_true
        dW2 = np.dot(a1.T, dZ2) / number_of_samples
        db2 = np.sum(dZ2, axis=0, keepdims=True) / number_of_samples

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self._hidden_activation_derivative(z1, a1)
        dW1 = np.dot(X.T, dZ1) / number_of_samples
        db1 = np.sum(dZ1, axis=0, keepdims=True) / number_of_samples

        gradients = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2,
        }
        return gradients

    def update_parameters(self, gradients: dict[str, np.ndarray]) -> None:
        """Update weights and biases using gradient descent."""
        self.W1 -= self.learning_rate * gradients["dW1"]
        self.b1 -= self.learning_rate * gradients["db1"]
        self.W2 -= self.learning_rate * gradients["dW2"]
        self.b2 -= self.learning_rate * gradients["db2"]

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 100,
        print_every: int = 1,
    ) -> dict[str, list[float]]:
        """Train the neural network using full-batch gradient descent."""
        if epochs <= 0:
            raise ValueError("Epochs must be greater than 0.")

        if print_every <= 0:
            raise ValueError("print_every must be greater than 0.")

        history = {"loss": [], "accuracy": []}

        for epoch in range(1, epochs + 1):
            predictions, cache = self.forward(X_train)
            loss = self.compute_loss(y_train, predictions)
            accuracy = self._calculate_accuracy_from_one_hot(y_train, predictions)

            gradients = self.backward(y_train, cache)
            self.update_parameters(gradients)

            history["loss"].append(loss)
            history["accuracy"].append(accuracy)

            if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
                print(
                    f"Epoch {epoch:03d}/{epochs} | "
                    f"Loss: {loss:.4f} | Accuracy: {accuracy:.4f}"
                )

        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        predictions, _ = self.forward(X)
        return predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
