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
    
    train_predictions = nn.predict(X_train).flatten()
    train_pred = (train_predictions > 0.5).astype(int)
    train_accuracy = compute_binary_accuracy(train_pred, y_train_mapped)
    train_loss = -np.mean(y_train_mapped * np.log(np.clip(train_predictions, 1e-15, 1 - 1e-15)))
    
    predictions = nn.predict(X_test).flatten()
    y_pred = (predictions > 0.5).astype(int)
    accuracy = compute_binary_accuracy(y_pred, y_test_mapped)
    test_loss = -np.mean(y_test_mapped * np.log(np.clip(predictions, 1e-15, 1 - 1e-15)))
    
    print(f"\n{'='*40}")
    print("Performance Evaluation")
    print(f"{'='*40}")
    print(f"Training Set - Accuracy: {train_accuracy * 100:.2f}%, Loss: {train_loss:.4f}")
    print(f"Test Set     - Accuracy: {accuracy * 100:.2f}%, Loss: {test_loss:.4f}")
    print(f"{'='*40}")
    
    return nn, accuracy, losses, y_pred, train_accuracy, test_loss, train_loss


def compare_model_structures(X_train, y_train, X_test, y_test, class_pair):
    print(f"\n{'='*60}")
    print(f"Comparing Model Structures for {class_pair[0]} vs {class_pair[1]}")
    print(f"{'='*60}")
    
    structures = [
        [32],
        [128],
        [512],
        [32, 32],
        [128, 64],
    ]
    
    results = []
    for hidden_layers in structures:
        nn, accuracy, losses, y_pred, train_acc, test_loss, train_loss = train_binary_classifier(
            X_train, y_train, X_test, y_test, class_pair, 
            hidden_layers=hidden_layers, epochs=200, lr=0.01
        )
        results.append({
            'structure': hidden_layers,
            'accuracy': accuracy,
            'train_accuracy': train_acc,
            'final_loss': test_loss,
            'train_loss': train_loss,
            'epochs_to_converge': len(losses)
        })
    
    print(f"\n{'='*60}")
    print("Model Structure Comparison Summary")
    print(f"{'='*60}")
    print(f"{'Structure':<20} {'Accuracy':<12} {'Final Loss':<12} {'Epochs':<10}")
    print("-" * 60)
    for r in results:
        struct_str = str(r['structure']).replace(' ', '')
        print(f"{struct_str:<20} {r['accuracy']*100:.2f}%      {r['final_loss']:.4f}      {r['epochs_to_converge']}")
    
    return results


def plot_structure_comparison(results_list, class_pair):
    structures = [r['structure'] for r in results_list]
    train_accuracies = [r['train_accuracy'] * 100 for r in results_list]
    test_accuracies = [r['accuracy'] * 100 for r in results_list]
    train_losses = [r.get('train_loss', r['final_loss']) for r in results_list]
    test_losses = [r['final_loss'] for r in results_list]
    
    struct_labels = [str(s).replace(' ', '') for s in structures]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    x = range(len(structures))
    
    axes[0, 0].bar(x, train_accuracies, color='steelblue', alpha=0.7)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(struct_labels)
    axes[0, 0].set_xlabel('Model Structure (Hidden Layers)')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title(f'Training Set Accuracy ({class_pair[0]} and {class_pair[1]})')
    axes[0, 0].set_ylim([90, 100])
    axes[0, 0].grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(train_accuracies):
        axes[0, 0].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=9)
    
    axes[0, 1].bar(x, test_accuracies, color='seagreen', alpha=0.7)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(struct_labels)
    axes[0, 1].set_xlabel('Model Structure (Hidden Layers)')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title(f'Test Set Accuracy ({class_pair[0]} and {class_pair[1]})')
    axes[0, 1].set_ylim([90, 100])
    axes[0, 1].grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(test_accuracies):
        axes[0, 1].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=9)
    
    axes[1, 0].bar(x, train_losses, color='coral', alpha=0.7)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(struct_labels)
    axes[1, 0].set_xlabel('Model Structure (Hidden Layers)')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title(f'Training Set Loss ({class_pair[0]} and {class_pair[1]})')
    axes[1, 0].grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(train_losses):
        axes[1, 0].text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=9)
    
    axes[1, 1].bar(x, test_losses, color='mediumpurple', alpha=0.7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(struct_labels)
    axes[1, 1].set_xlabel('Model Structure (Hidden Layers)')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title(f'Test Set Loss ({class_pair[0]} and {class_pair[1]})')
    axes[1, 1].grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(test_losses):
        axes[1, 1].text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=9)
    
    plt.rcParams['font.style'] = 'normal'
    for ax in axes.flat:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontstyle('normal')
    
    plt.tight_layout()
    plt.savefig(f'structure_comparison_{class_pair[0]}_vs_{class_pair[1]}.png', dpi=150)
    plt.show()


def plot_prediction_examples(X_test, y_test_binary, y_pred, class_pair, num_samples=8):
    correct_mask = (y_pred == y_test_binary)
    correct_indices = np.where(correct_mask)[0][:num_samples]
    incorrect_indices = np.where(~correct_mask)[0][:num_samples]
    
    total_incorrect = np.sum(~correct_mask)
    print(f"Correct predictions: {len(correct_indices)}, Incorrect: {total_incorrect}")
    
    num_cols = num_samples
    num_rows = 2
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 1.5, 4))
    
    for i, idx in enumerate(correct_indices):
        axes[0, i].imshow(X_test[idx].reshape(28, 28), cmap='gray')
        true_label = class_pair[y_test_binary[idx]]
        pred_label = class_pair[y_pred[idx]]
        axes[0, i].set_title(f'True: {true_label}\nPred: {pred_label}', fontsize=10)
        axes[0, i].axis('off')
    
    for i in range(num_cols):
        if i < len(incorrect_indices):
            idx = incorrect_indices[i]
            axes[1, i].imshow(X_test[idx].reshape(28, 28), cmap='gray')
            true_label = class_pair[y_test_binary[idx]]
            pred_label = class_pair[y_pred[idx]]
            axes[1, i].set_title(f'True: {true_label}\nPred: {pred_label}', fontsize=10, color='red')
            axes[1, i].axis('off')
        else:
            axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Correct', fontsize=11, rotation=0, ha='right', va='center')
    axes[1, 0].set_ylabel('Incorrect', fontsize=11, rotation=0, ha='right', va='center', color='red')
    
    plt.suptitle(f'Prediction Examples: {class_pair[0]} vs {class_pair[1]}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'prediction_examples_{class_pair[0]}_vs_{class_pair[1]}.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    """
    P3Q1: Binary Classification with Neural Network
    
    Objective: Train binary classifiers to distinguish between digit pairs
    Dataset: MNIST (subset), 5000 samples per class pair
    
    Class pairs tested:
        - (1, 7): visually similar digits
        - (3, 5): visually similar digits
        - (4, 9): visually similar digits
    
    Workflow for each class pair:
        1. Load dataset filtered by target labels
        2. Split into train (80%) and test (20%)
        3. Compare different network structures:
           - [64], [128], [256] (single layer)
           - [64, 32], [128, 64], [256, 128] (two layers)
        4. Train best model with hidden_layers=[128, 64], epochs=200
        5. Visualize correct and incorrect predictions
    
    Output: 
        - structure_comparison_*.png (per class pair)
        - all_structure_comparison.png (combined)
        - prediction_examples_*.png
    """
    
    class_pairs = [(1, 7), (3, 5), (4, 9)]
    
    print("Loading full MNIST dataset...")

    
    all_results = []
    
    for class_pair in class_pairs:
        print(f"\n{'#'*60}")
        print(f"# Processing: {class_pair[0]} vs {class_pair[1]}")
        print(f"{'#'*60}")

        X_all, y_all, labels_all = load_dataset("mnist.csv", target_labels=class_pair)
        X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(X_all, y_all, labels_all, test_ratio=0.2)
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        results = compare_model_structures(X_train, y_train, X_test, y_test, class_pair)
        all_results.append((class_pair, results))
        
        plot_structure_comparison(results, class_pair)
        
        nn_best, accuracy, losses, y_pred, train_acc, test_loss, train_loss = train_binary_classifier(
            X_train, y_train, X_test, y_test, 
            class_pair, hidden_layers=[128, 64], epochs=200
        )
        
        y_test_labels = np.argmax(y_test, axis=1)
        label_map = {class_pair[0]: 0, class_pair[1]: 1}
        y_test_binary = np.array([label_map[l] for l in y_test_labels])
        
        plot_prediction_examples(X_test, y_test_binary, y_pred, class_pair, num_samples=8)
    
    plt.figure(figsize=(12, 8))
    for idx, (class_pair, results) in enumerate(all_results):
        plt.subplot(1, 3, idx + 1)
        struct_labels = [str(r['structure']).replace(' ', '') for r in results]
        accuracies = [r['accuracy'] * 100 for r in results]
        plt.bar(range(len(struct_labels)), accuracies, color='steelblue', alpha=0.8)
        plt.xticks(range(len(struct_labels)), struct_labels, rotation=45, ha='right')
        plt.ylabel('Accuracy (%)')
        plt.title(f'{class_pair[0]} vs {class_pair[1]}')
        plt.ylim([80, 100])
    plt.suptitle('Model Structure vs Accuracy (All Class Pairs)', fontsize=14)
    plt.tight_layout()
    plt.savefig('all_structure_comparison.png', dpi=150)
    plt.show()
    