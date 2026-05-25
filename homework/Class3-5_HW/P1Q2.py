import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs


def smiles_to_fingerprint(smiles, fp_type='morgan', radius=3, fpSize=128):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(fpSize, dtype=int)
    if fp_type == 'morgan':
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)
    elif fp_type == 'rdkit':
        generator = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=fpSize)
    elif fp_type == 'atompair':
        generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=fpSize)
    elif fp_type == 'topological':
        generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=fpSize)
    else:
        raise ValueError(f"不支持的指纹类型: {fp_type}")

    fp = generator.GetFingerprint(mol)
    arr = np.zeros((fpSize,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def load_data(filepath, fp_type='morgan', radius=3, fpSize=128):
    df = pd.read_excel(filepath)
    X = []
    Y = []
    for i in range(len(df)):
        smiles = df.iloc[i]['COMPOUND_SMILES']
        finger = smiles_to_fingerprint(smiles, fp_type=fp_type, radius=radius, fpSize=fpSize)
        H_data = df.iloc[i]['H']
        ea_data = df.iloc[i]['EA']
        dcm_data = df.iloc[i]['DCM']
        meoh_data = df.iloc[i]['MeOH']
        et2o_data = df.iloc[i]['Et2O']
        x = np.concatenate([finger, [H_data], [ea_data], [dcm_data], [meoh_data], [et2o_data]])
        y = df.iloc[i]['Rf']
        X.append(x)
        Y.append(y)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def linear(x):
    return x


def linear_derivative(x):
    return np.ones_like(x)


class MBGDNet:
    """
    使用纯NumPy实现的神经网络，支持MBGD优化器
    损失函数: MSE
    支持激活函数: relu, sigmoid, linear
    """
    def __init__(self, layers_config, output_activation='sigmoid'):
        """
        layers_config: [(input_dim, 64, 'relu'), (64, 32, 'relu'), (32, 1, 'sigmoid')]
        output_activation: 'sigmoid' 或 'linear'
        """
        self.layers_config = layers_config
        self.output_activation = output_activation
        self.weights = []
        self.biases = []
        self.activations = []
        self.z_cache = []  # 存储线性变换结果用于反向传播
        self.a_cache = []  # 存储激活结果

        # 初始化权重和偏置
        for i, (in_dim, out_dim, activation) in enumerate(layers_config):
            # Xavier初始化
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            self.weights.append(np.random.uniform(-limit, limit, (in_dim, out_dim)))
            self.biases.append(np.zeros((1, out_dim)))
            self.activations.append(activation)

    def _get_activation(self, name):
        if name == 'relu':
            return relu, relu_derivative
        elif name == 'sigmoid':
            return sigmoid, sigmoid_derivative
        elif name == 'linear':
            return linear, linear_derivative
        else:
            raise ValueError(f"不支持的激活函数: {name}")

    def forward(self, X):
        """前向传播"""
        self.z_cache = []
        self.a_cache = [X]
        a = X

        for i, (w, b, act_name) in enumerate(zip(self.weights, self.biases, self.activations)):
            z = np.dot(a, w) + b
            self.z_cache.append(z)
            act_func, _ = self._get_activation(act_name)
            a = act_func(z)
            self.a_cache.append(a)

        return a

    def backward(self, X, y, learning_rate):
        """反向传播并更新参数"""
        m = X.shape[0]
        y = y.reshape(-1, 1)

        # 输出层梯度: dL/dz = dL/da * da/dz
        # MSE损失: L = 0.5 * (a - y)^2, dL/da = (a - y)
        a_last = self.a_cache[-1]
        z_last = self.z_cache[-1]

        if self.activations[-1] == 'sigmoid':
            # a = sigmoid(z), da/dz = sigmoid(z) * (1 - sigmoid(z)) = a * (1 - a)
            delta = (a_last - y) * a_last * (1 - a_last)
        elif self.activations[-1] == 'linear':
            # a = z, da/dz = 1
            delta = (a_last - y)
        else:
            _, deriv_func = self._get_activation(self.activations[-1])
            delta = (a_last - y) * deriv_func(z_last)

        # 存储梯度
        dw_list = []
        db_list = []

        # 从最后一层向前传播
        for i in reversed(range(len(self.weights))):
            dw = np.dot(self.a_cache[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m

            dw_list.insert(0, dw)
            db_list.insert(0, db)

            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                _, deriv_func = self._get_activation(self.activations[i-1])
                delta = delta * deriv_func(self.z_cache[i-1])

        # 更新参数
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dw_list[i]
            self.biases[i] -= learning_rate * db_list[i]

    def train(self, X_train, y_train, X_val, y_val, epochs=500, batch_size=32,
              learning_rate=0.01, patience=30):
        """
        使用MBGD训练模型

        Parameters:
        -----------
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        epochs: 训练轮数
        batch_size: mini-batch大小
        learning_rate: 初始学习率
        patience: 早停耐心值
        """
        m = X_train.shape[0]
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        best_biases = None
        learning_rate = learning_rate

        for epoch in range(epochs):
            # 打乱数据
            indices = np.random.permutation(m)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0
            num_batches = 0

            # Mini-batch训练
            for i in range(0, m, batch_size):
                end_idx = min(i + batch_size, m)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]

                # 前向传播
                y_pred = self.forward(X_batch)

                # 计算损失
                batch_loss = np.mean((y_pred.flatten() - y_batch) ** 2)
                epoch_loss += batch_loss
                num_batches += 1

                # 反向传播 - 使用学习率调度
                self.backward(X_batch, y_batch, learning_rate)

            epoch_loss /= num_batches
            train_losses.append(epoch_loss)

            # 验证
            val_pred = self.predict(X_val)
            val_loss = np.mean((val_pred - y_val) ** 2)
            val_losses.append(val_loss)


            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳权重
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # 恢复最佳权重
        if best_weights is not None:
            self.weights = best_weights
            self.biases = best_biases

        return train_losses, val_losses

    def predict(self, X):
        """预测"""
        return self.forward(X).flatten()

    def evaluate(self, X, y):
        """评估模型"""
        y_pred = self.predict(X)
        mse = np.mean((y_pred - y) ** 2)
        rmse = np.sqrt(mse)
        # R^2 score
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return y_pred, r2, mse, rmse

    def get_params_count(self):
        """获取参数数量"""
        total = 0
        for w, b in zip(self.weights, self.biases):
            total += w.size + b.size
        return total


def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    """划分数据集"""
    n = len(X)
    indices = np.random.permutation(n)

    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])


def create_model_config(input_dim, hidden_layers=[64, 32], output_activation='sigmoid'):
    """
    创建模型配置
    """
    layers_config = []
    prev_dim = input_dim

    for hidden_dim in hidden_layers:
        layers_config.append((prev_dim, hidden_dim, 'relu'))
        prev_dim = hidden_dim

    layers_config.append((prev_dim, 1, output_activation))
    return layers_config


def main():
    # 设置随机种子
    np.random.seed(42)

    # 加载数据
    print("Loading data...")
    X, Y = load_data('TLC_dataset.xlsx', fp_type='morgan', radius=3, fpSize=128)
    print(f"Data shape: X={X.shape}, Y={Y.shape}")

    # 划分数据集
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, Y)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ========== 示例1: Sigmoid输出激活 ==========
    print("\n" + "="*50)
    print("Model with Sigmoid Output Activation")
    print("="*50)

    layers_config_sigmoid = [
        (X.shape[1], 64, 'relu'),
        (64, 32, 'relu'),
        (32, 1, 'sigmoid')
    ]

    model_sigmoid = MBGDNet(layers_config_sigmoid, output_activation='sigmoid')
    print(f"Total parameters: {model_sigmoid.get_params_count()}")

    print("\nTraining...")
    train_losses_sigmoid, val_losses_sigmoid = model_sigmoid.train(
        X_train, y_train, X_val, y_val,
        epochs=1000, batch_size=32, learning_rate=0.1
    )

    y_pred_sigmoid, r2_sigmoid, mse_sigmoid, rmse_sigmoid = model_sigmoid.evaluate(X_test, y_test)
    print(f"\nSigmoid Model - Test R2: {r2_sigmoid:.4f}, MSE: {mse_sigmoid:.6f}, RMSE: {rmse_sigmoid:.6f}")

    # ========== 示例2: Linear输出激活 ==========
    print("\n" + "="*50)
    print("Model with Linear Output Activation")
    print("="*50)

    layers_config_linear = [
        (X.shape[1], 64, 'relu'),
        (64, 32, 'relu'),
        (32, 1, 'linear')
    ]

    model_linear = MBGDNet(layers_config_linear, output_activation='linear')
    print(f"Total parameters: {model_linear.get_params_count()}")

    print("\nTraining...")
    train_losses_linear, val_losses_linear = model_linear.train(
        X_train, y_train, X_val, y_val,
        epochs=1000, batch_size=32, learning_rate=0.1
    )

    y_pred_linear, r2_linear, mse_linear, rmse_linear = model_linear.evaluate(X_test, y_test)
    print(f"\nLinear Model - Test R2: {r2_linear:.4f}, MSE: {mse_linear:.6f}, RMSE: {rmse_linear:.6f}")

    # 绘制对比图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Sigmoid - Training History
    axes[0, 0].plot(train_losses_sigmoid, label='Train Loss')
    axes[0, 0].plot(val_losses_sigmoid, label='Val Loss')
    axes[0, 0].set_title('Sigmoid: Training History')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Sigmoid - Prediction vs Actual
    axes[0, 1].scatter(y_test, y_pred_sigmoid, alpha=0.5, edgecolors='black', linewidth=0.5)
    axes[0, 1].plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
    axes[0, 1].set_title(f'Sigmoid: Prediction vs Actual (R2={r2_sigmoid:.4f})')
    axes[0, 1].set_xlabel('Actual Rf')
    axes[0, 1].set_ylabel('Predicted Rf')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)

    # Linear - Training History
    axes[1, 0].plot(train_losses_linear, label='Train Loss')
    axes[1, 0].plot(val_losses_linear, label='Val Loss')
    axes[1, 0].set_title('Linear: Training History')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MSE Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Linear - Prediction vs Actual
    axes[1, 1].scatter(y_test, y_pred_linear, alpha=0.5, edgecolors='black', linewidth=0.5)
    axes[1, 1].plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
    axes[1, 1].set_title(f'Linear: Prediction vs Actual (R2={r2_linear:.4f})')
    axes[1, 1].set_xlabel('Actual Rf')
    axes[1, 1].set_ylabel('Predicted Rf')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('q2_comparison.png', dpi=300)
    plt.show()

    print("\nResults saved to q2_comparison.png")


if __name__ == "__main__":
    main()
