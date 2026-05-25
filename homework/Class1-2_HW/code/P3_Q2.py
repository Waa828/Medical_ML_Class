import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

np.random.seed(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def softmax_derivative(x):
    s = softmax(x)
    return s * (1 - s)


class FullyConnectedLayer():
    def __init__(self, input_dim, output_dim, activation='relu', is_output_layer=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.is_output_layer = is_output_layer
        
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.bias = np.zeros((1, output_dim))
        
        self.A_in = None
        self.Z = None
        self.A_out = None
        self.y_true = None

    def forward(self, A_in):
        self.A_in = A_in
        self.Z = np.dot(A_in, self.weights) + self.bias
        
        if self.activation == 'sigmoid':
            self.A_out = sigmoid(self.Z)
        elif self.activation == 'relu':
            self.A_out = relu(self.Z)
        elif self.activation == 'softmax':
            self.A_out = softmax(self.Z)
        else:
            self.A_out = self.Z
        
        return self.A_out

    def backward(self, dA_out=None, y_true=None, is_output_layer=False):
        if is_output_layer and y_true is not None:
            self.y_true = y_true
            if self.activation == 'sigmoid':
                dZ = self.A_out - self.y_true
            elif self.activation == 'softmax':
                dZ = self.A_out - self.y_true
            else:
                dZ = dA_out * relu_derivative(self.Z)
        else:
            dZ = dA_out * relu_derivative(self.Z)
        
        m = self.A_in.shape[0]
        dW = np.dot(self.A_in.T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        
        dA_in = np.dot(dZ, self.weights.T)
        
        return dA_in, dW, db

    def print_dims(self, label, *args):
        print(f"{label}: " + ", ".join([str(a.shape) for a in args]))


class NeuralNetwork:
    def __init__(self, layers_config):
        self.layers = []
        for config in layers_config:
            input_dim, output_dim, activation = config
            is_output = (config == layers_config[-1])
            self.layers.append(FullyConnectedLayer(input_dim, output_dim, activation, is_output))

    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, y_true):
        grads = []
        num_layers = len(self.layers)
        
        for i, layer in enumerate(reversed(range(num_layers))):
            layer_idx = num_layers - 1 - i
            is_last = (layer_idx == num_layers - 1)
            
            if is_last:
                dA = None
                dA_in, dW, db = self.layers[layer_idx].backward(dA_out=None, y_true=y_true, is_output_layer=True)
            else:
                next_layer = self.layers[layer_idx + 1]
                dA = next_layer.dA_in if hasattr(next_layer, 'dA_in') else None
                dA_in, dW, db = self.layers[layer_idx].backward(dA_out=dA)
            
            self.layers[layer_idx].dA_in = dA_in
            grads.append((dW, db))
        
        grads.reverse()
        return grads

    def update_params(self, grads, lr=0.001):
        for layer, (dW, db) in zip(self.layers, grads):
            layer.weights -= lr * dW
            layer.bias -= lr * db

    def train(self, X, y, epochs, learning_rate):
        losses = []
        for epoch in range(epochs):
            A = self.forward(X)
            
            loss = -np.mean(y * np.log(np.clip(A, 1e-15, 1 - 1e-15)))
            losses.append(loss)
            
            grads = self.backward(y_true=y)
            self.update_params(grads, learning_rate)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
        
        return losses

    def predict(self, X):
        return self.forward(X)

    def save_model(self, filepath):
        model_data = {
            'layers_config': [],
            'weights': [],
            'biases': []
        }
        for layer in self.layers:
            model_data['layers_config'].append({
                'input_dim': layer.input_dim,
                'output_dim': layer.output_dim,
                'activation': layer.activation,
                'is_output_layer': layer.is_output_layer
            })
            model_data['weights'].append(layer.weights)
            model_data['biases'].append(layer.bias)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        layers_config = [
            (cfg['input_dim'], cfg['output_dim'], cfg['activation'])
            for cfg in model_data['layers_config']
        ]
        
        nn = NeuralNetwork(layers_config)
        
        for i, layer in enumerate(nn.layers):
            layer.weights = model_data['weights'][i]
            layer.bias = model_data['biases'][i]
        
        print(f"Model loaded from {filepath}")
        return nn


def compute_binary_accuracy(y_pred, y_true):
    y_pred = (y_pred > 0.5).astype(int).flatten()
    y_true = y_true.flatten()
    return np.mean(y_pred == y_true)


def compute_multi_accuracy(y_pred, y_true):
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_true, axis=1)
    accuracy = np.mean(y_pred_labels == y_true_labels)
    return accuracy


def binary_cross_entropy_loss(y_pred, y_true):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


def softmax_loss(y_pred, y_true):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def plot_image(image, label):
    """
    Plots a single MNIST image (28x28) and shows its numeric label
    derived from the one-hot vector.
    """
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f"Label: {np.argmax(label)}")  # Convert one-hot to numeric
    plt.axis('off')
    plt.show()


def load_dataset(csv_path: str,
                 target_labels = None,
                 shuffles: bool = True,
                 num: int | None = None,
                 ):
    """
    Reads MNIST data from CSV, normalizes, one-hot encodes,
    shuffles, and splits it into train/valid/test sets.
    
    Optionally, only returns certain labels if target_labels is provided.
    target_labels = [1,3,5]
    """
    df = pd.read_csv(csv_path, header=None)
    labels = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.astype(np.float32) / 255.0

    if target_labels is not None:
        mask = np.isin(labels, target_labels)
        X = X[mask]
        labels = labels[mask]

    if shuffles:
        num_samples = X.shape[0]
        indices = np.random.permutation(num_samples)
        X = X[indices]
        labels = labels[indices]

    num_classes = 10
    y = np.eye(num_classes)[labels]

    if num is not None:
        num_samples = X.shape[0]
        if num <= num_samples:
            X = X[:num]
            y = y[:num]
        else:
            raise ValueError(f"Requested number of samples ({num}) exceeds the total number of available samples ({num_samples}).")

    return X, y, labels


def train_test_split(X: np.ndarray, y: np.ndarray, labels_all: np.ndarray, test_ratio: float = 0.2):
    if not isinstance(test_ratio, (int, float)):
        raise TypeError("test_ratio must be a number")
    if test_ratio <= 0 or test_ratio >= 1:
        raise ValueError("test_ratio must be in range (0, 1)")
    
    num_samples = X.shape[0]
    test_size = int(num_samples * test_ratio)
    
    indices = np.random.permutation(num_samples)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    label_train = labels_all[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    label_test = labels_all[test_indices]

    return X_train, X_test, y_train, y_test, label_train, label_test


def load_binary_dataset(csv_path: str, class_pair: tuple, num: int | None = None):
    df = pd.read_csv(csv_path, header=None)
    labels = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.astype(np.float32) / 255.0
    
    mask = np.isin(labels, class_pair)
    X = X[mask]
    labels = labels[mask]
    
    label_map = {class_pair[0]: 0, class_pair[1]: 1}
    binary_labels = np.array([label_map[l] for l in labels])
    
    indices = np.random.permutation(len(X))
    X = X[indices]
    binary_labels = binary_labels[indices]
    
    if num is not None:
        X = X[:num]
        binary_labels = binary_labels[:num]
    
    return X, binary_labels


def plot_binary_predictions(X, y_true, y_pred, class_pair, num_samples=5):
    correct_mask = (y_pred == y_true)
    incorrect_mask = ~correct_mask
    
    correct_indices = np.where(correct_mask)[0][:num_samples]
    incorrect_indices = np.where(incorrect_mask)[0][:num_samples]
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i, idx in enumerate(correct_indices):
        axes[0, i].imshow(X[idx].reshape(28, 28), cmap='gray')
        axes[0, i].set_title(f"True: {class_pair[y_true[idx]]}, Pred: {class_pair[y_pred[idx]]}")
        axes[0, i].axis('off')
    
    for i, idx in enumerate(incorrect_indices):
        axes[1, i].imshow(X[idx].reshape(28, 28), cmap='gray')
        axes[1, i].set_title(f"True: {class_pair[y_true[idx]]}, Pred: {class_pair[y_pred[idx]]}")
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Correct', fontsize=12)
    axes[1, 0].set_ylabel('Incorrect', fontsize=12)
    plt.suptitle(f"Binary Classification: {class_pair[0]} vs {class_pair[1]}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f'predictions_{class_pair[0]}_vs_{class_pair[1]}.png', dpi=150)
    plt.show()


def train_binary_classifier(X_train, y_train, X_test, y_test, class_pair, hidden_layers=[128, 64], epochs=50, lr=0.01):
    print(f"\n{'='*50}")
    print(f"Training classifier: {class_pair[0]} vs {class_pair[1]}")
    print(f"{'='*50}")
    
    y_train_labels = np.argmax(y_train, axis=1).reshape(-1, 1)
    y_test_labels = np.argmax(y_test, axis=1).reshape(-1, 1)
    
    label_map = {class_pair[0]: 0, class_pair[1]: 1}
    y_train_mapped = np.array([[label_map[l[0]]] for l in y_train_labels])
    y_test_mapped = np.array([[label_map[l[0]]] for l in y_test_labels])
    
    layers_config = [(784, hidden_layers[0], 'relu')]
    for i in range(len(hidden_layers) - 1):
        layers_config.append((hidden_layers[i], hidden_layers[i+1], 'relu'))
    layers_config.append((hidden_layers[-1], 1, 'sigmoid'))
    
    nn = NeuralNetwork(layers_config)
    
    print(f"Model structure: 784 -> {' -> '.join(map(str, hidden_layers))} -> 1")
    
    losses = nn.train(X_train, y_train_mapped, epochs=epochs, learning_rate=lr)
    
    predictions = nn.predict(X_test).flatten()
    y_pred = (predictions > 0.5).astype(int)
    accuracy = compute_binary_accuracy(y_pred, y_test_mapped)
    
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    print(f"Final Loss: {losses[-1]:.4f}")
    
    return nn, accuracy, losses, y_pred


def plot_multi_predictions(X, y_true, y_pred, num_samples=5):
    correct_mask = (y_pred == y_true)
    incorrect_mask = ~correct_mask
    
    correct_indices = np.where(correct_mask)[0][:num_samples]
    incorrect_indices = np.where(incorrect_mask)[0][:num_samples]
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i, idx in enumerate(correct_indices):
        axes[0, i].imshow(X[idx].reshape(28, 28), cmap='gray')
        axes[0, i].set_title(f"True: {y_true[idx]}, Pred: {y_pred[idx]}")
        axes[0, i].axis('off')
    
    for i, idx in enumerate(incorrect_indices):
        axes[1, i].imshow(X[idx].reshape(28, 28), cmap='gray')
        axes[1, i].set_title(f"True: {y_true[idx]}, Pred: {y_pred[idx]}")
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Correct', fontsize=12)
    axes[1, 0].set_ylabel('Incorrect', fontsize=12)
    plt.suptitle(f"Multi-class Classification (10 classes)", fontsize=14)
    plt.tight_layout()
    plt.savefig('multi_class_predictions.png', dpi=150)
    plt.show()


def train_multi_classifier(X_train, y_train, X_test, y_test, hidden_layers=[256, 128], epochs=100, lr=0.01):
    print(f"\n{'='*50}")
    print(f"Training 10-class Classifier")
    print(f"{'='*50}")
    
    layers_config = [(784, hidden_layers[0], 'relu')]
    for i in range(len(hidden_layers) - 1):
        layers_config.append((hidden_layers[i], hidden_layers[i+1], 'relu'))
    layers_config.append((hidden_layers[-1], 10, 'softmax'))
    
    nn = NeuralNetwork(layers_config)
    
    print(f"Model structure: 784 -> {' -> '.join(map(str, hidden_layers))} -> 10")
    
    losses = nn.train(X_train, y_train, epochs=epochs, learning_rate=lr)
    
    train_predictions = nn.predict(X_train)
    train_pred_labels = np.argmax(train_predictions, axis=1)
    train_true_labels = np.argmax(y_train, axis=1)
    train_accuracy = compute_multi_accuracy(train_predictions, y_train)
    train_loss = -np.mean(np.sum(y_train * np.log(np.clip(train_predictions, 1e-15, 1 - 1e-15)), axis=1))
    
    predictions = nn.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    accuracy = compute_multi_accuracy(predictions, y_test)
    test_loss = -np.mean(np.sum(y_test * np.log(np.clip(predictions, 1e-15, 1 - 1e-15)), axis=1))
    
    print(f"\n{'='*40}")
    print("Performance Evaluation")
    print(f"{'='*40}")
    print(f"Training Set - Accuracy: {train_accuracy * 100:.2f}%, Loss: {train_loss:.4f}")
    print(f"Test Set     - Accuracy: {accuracy * 100:.2f}%, Loss: {test_loss:.4f}")
    print(f"{'='*40}")
    
    return nn, accuracy, losses, y_pred, train_accuracy, test_loss, train_loss


def compare_training_params(X_train, y_train, X_test, y_test):
    print(f"\n{'='*60}")
    print("Comparing Training Parameters")
    print(f"{'='*60}")
    
    learning_rates = [0.01, 0.05, 0.1]
    results_lr = []
    
    print("\n--- Testing Learning Rates ---")
    for lr in learning_rates:
        print(f"\nTesting learning rate: {lr}")
        nn, acc, losses, y_pred, train_acc, test_loss, train_loss = train_multi_classifier(
            X_train, y_train, X_test, y_test,
            hidden_layers=[256, 128], epochs=100, lr=lr
        )
        results_lr.append({'lr': lr, 'train_acc': train_acc, 'test_acc': acc, 'train_loss': train_loss, 'test_loss': test_loss})
    
    epochs_list = [50, 100, 200]
    results_epochs = []
    
    print("\n--- Testing Epochs ---")
    for ep in epochs_list:
        print(f"\nTesting epochs: {ep}")
        nn, acc, losses, y_pred, train_acc, test_loss, train_loss = train_multi_classifier(
            X_train, y_train, X_test, y_test,
            hidden_layers=[256, 128], epochs=ep, lr=0.05
        )
        results_epochs.append({'epochs': ep, 'train_acc': train_acc, 'test_acc': acc, 'train_loss': train_loss, 'test_loss': test_loss})
    
    structures = [[128], [256, 128], [512, 256]]
    results_struct = []
    
    print("\n--- Testing Model Structures ---")
    for struct in structures:
        print(f"\nTesting structure: {struct}")
        nn, acc, losses, y_pred, train_acc, test_loss, train_loss = train_multi_classifier(
            X_train, y_train, X_test, y_test,
            hidden_layers=struct, epochs=100, lr=0.05
        )
        results_struct.append({'structure': struct, 'train_acc': train_acc, 'test_acc': acc, 'train_loss': train_loss, 'test_loss': test_loss})
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    lrs = [r['lr'] for r in results_lr]
    axes[0].plot(lrs, [r['train_acc']*100 for r in results_lr], 'o-', label='Train', color='steelblue')
    axes[0].plot(lrs, [r['test_acc']*100 for r in results_lr], 's-', label='Test', color='seagreen')
    axes[0].set_xlabel('Learning Rate')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Learning Rate vs Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    eps = [r['epochs'] for r in results_epochs]
    axes[1].plot(eps, [r['train_acc']*100 for r in results_epochs], 'o-', label='Train', color='steelblue')
    axes[1].plot(eps, [r['test_acc']*100 for r in results_epochs], 's-', label='Test', color='seagreen')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Epochs vs Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    struct_labels = [str(r['structure']).replace(' ', '') for r in results_struct]
    x = range(len(struct_labels))
    width = 0.35
    axes[2].bar([i - width/2 for i in x], [r['train_acc']*100 for r in results_struct], width, label='Train', color='steelblue')
    axes[2].bar([i + width/2 for i in x], [r['test_acc']*100 for r in results_struct], width, label='Test', color='seagreen')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(struct_labels)
    axes[2].set_xlabel('Model Structure')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('Model Structure vs Accuracy')
    axes[2].legend()
    axes[2].grid(True, axis='y', alpha=0.3)
    
    plt.suptitle('Impact of Training Parameters on Accuracy', fontsize=14)
    plt.tight_layout()
    plt.savefig('training_params_comparison.png', dpi=150)
    plt.show()
    
    return results_lr, results_epochs, results_struct


if __name__ == '__main__':
    """
    P3Q2: Multi-class Classification (10-class) with Neural Network
    
    Objective: Train a neural network to classify all 10 digits (0-9)
    Dataset: MNIST, 5000 samples
    
    Network Architecture:
        - Input: 784 (28x28 flattened images)
        - Hidden layers: [256, 128] with ReLU activation
        - Output: 10 with softmax activation
    
    Workflow:
        1. Load 5000 MNIST samples
        2. Split into train (80%) and test (20%)
        3. Train multi-class classifier (epochs=200, lr=0.05)
        4. Plot training loss curve
        5. Visualize correct/incorrect predictions
        6. Compare training parameters (learning rate, epochs, structure)
    
    Hyperparameters:
        - epochs: 200
        - learning_rate: 0.05
        - hidden_layers: [256, 128]
    
    Output:
        - multi_class_loss.png
        - multi_class_predictions.png
        - training_params_comparison.png
    """
    print("="*60)
    print("10-Class Classification (MNIST)")
    print("="*60)
    
    X_all, y_all, labels_all = load_dataset("mnist.csv")
    X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(X_all, y_all, labels_all, test_ratio=0.2)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    nn, accuracy, losses, y_pred, train_acc, test_loss, train_loss = train_multi_classifier(
        X_train, y_train, X_test, y_test,
        hidden_layers=[256, 128], epochs=200, lr=0.05
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('10-Class Classification Loss')
    plt.grid(True)
    plt.savefig('multi_class_loss.png', dpi=150)
    plt.show()
    
    y_true_labels = np.argmax(y_test, axis=1)
    plot_multi_predictions(X_test, y_true_labels, y_pred, num_samples=5)
    
    print("\n" + "="*60)
    print("Parameter Impact Analysis")
    print("="*60)
    results_lr, results_epochs, results_struct = compare_training_params(X_train, y_train, X_test, y_test)
    
    print("\n" + "="*60)
    print("10-Class Classification Results")
    print("="*60)
    print(f"Final Accuracy: {accuracy * 100:.2f}%")
    