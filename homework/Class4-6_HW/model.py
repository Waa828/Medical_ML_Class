import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs


def smiles_to_fingerprint(smiles, fp_type='morgan', radius=4, fpSize=512):
    """
    将SMILES字符串转换为分子指纹
    默认使用Morgan指纹（ECFP），r=4, s=512
    """
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
    """
    加载数据，默认使用Morgan指纹（ECFP），r=3, s=128
    """
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
    支持MBGD优化器
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
        self.z_cache = []
        self.a_cache = []

        for in_dim, out_dim, activation in layers_config:
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

        for w, b, act_name in zip(self.weights, self.biases, self.activations):
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

        a_last = self.a_cache[-1]

        if self.activations[-1] == 'sigmoid':
            delta = (a_last - y) * a_last * (1 - a_last)
        elif self.activations[-1] == 'linear':
            delta = (a_last - y)
        else:
            _, deriv_func = self._get_activation(self.activations[-1])
            z_last = self.z_cache[-1]
            delta = (a_last - y) * deriv_func(z_last)

        dw_list = []
        db_list = []

        for i in reversed(range(len(self.weights))):
            dw = np.dot(self.a_cache[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m

            dw_list.insert(0, dw)
            db_list.insert(0, db)

            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                _, deriv_func = self._get_activation(self.activations[i - 1])
                delta = delta * deriv_func(self.z_cache[i - 1])

        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dw_list[i]
            self.biases[i] -= learning_rate * db_list[i]

    def train(self, X_train, y_train, X_val, y_val, epochs=500, batch_size=32,
              learning_rate=0.01, patience=30):
        """
        使用MBGD训练模型
        """
        m = X_train.shape[0]
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        best_biases = None

        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0
            num_batches = 0

            for i in range(0, m, batch_size):
                end_idx = min(i + batch_size, m)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]

                y_pred = self.forward(X_batch)
                batch_loss = np.mean((y_pred.flatten() - y_batch) ** 2)
                epoch_loss += batch_loss
                num_batches += 1

                self.backward(X_batch, y_batch, learning_rate)

            epoch_loss /= num_batches
            train_losses.append(epoch_loss)

            val_pred = self.predict(X_val)
            val_loss = np.mean((val_pred - y_val) ** 2)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

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

    def save(self, filepath):
        """
        保存模型为.pkl文件
        """
        model_data = {
            'layers_config': self.layers_config,
            'output_activation': self.output_activation,
            'weights': self.weights,
            'biases': self.biases
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to: {filepath}")

    @staticmethod
    def load(filepath):
        """
        从.pkl文件加载模型
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        model = MBGDNet(model_data['layers_config'], output_activation=model_data['output_activation'])
        model.weights = model_data['weights']
        model.biases = model_data['biases']
        print(f"Model loaded from: {filepath}")
        return model


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
    np.random.seed(42)

    data_path = 'TLC_dataset.xlsx'
    fp_type = 'morgan'
    radius = 4
    fpSize = 512
    hidden_layers = [128, 64]
    output_activation = 'sigmoid'
    epochs = 500
    batch_size = 32
    learning_rate = 0.05
    patience = 30
    save_path = 'tlc_model.pkl'

    print("=" * 70)
    print("加载数据中...")
    print("=" * 70)
    X, y = load_data(data_path, fp_type=fp_type, radius=radius, fpSize=fpSize)
    print(f"数据集大小: {X.shape[0]} 样本, 特征维度: {X.shape[1]}")
    print(f"Rf 范围: [{y.min():.4f}, {y.max():.4f}], 均值: {y.mean():.4f}")

    print("\n" + "=" * 70)
    print("划分训练/验证/测试集 (70%/15%/15%)")
    print("=" * 70)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X, y, train_ratio=0.7, val_ratio=0.15
    )
    print(f"训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")

    print("\n" + "=" * 70)
    print("构建模型")
    print("=" * 70)
    input_dim = X.shape[1]
    layers_config = create_model_config(
        input_dim, hidden_layers=hidden_layers, output_activation=output_activation
    )
    model = MBGDNet(layers_config, output_activation=output_activation)
    print(f"网络结构: {layers_config}")
    print(f"参数总数: {model.get_params_count()}")

    print("\n" + "=" * 70)
    print("开始训练 (MBGD + 早停)")
    print("=" * 70)
    train_losses, val_losses = model.train(
        X_train, y_train, X_val, y_val,
        epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, patience=patience
    )

    print("\n" + "=" * 70)
    print("模型评估")
    print("=" * 70)
    _, train_r2, train_mse, train_rmse = model.evaluate(X_train, y_train)
    _, val_r2, val_mse, val_rmse = model.evaluate(X_val, y_val)
    _, test_r2, test_mse, test_rmse = model.evaluate(X_test, y_test)

    print(f"{'集合':<8}{'R²':>10}{'MSE':>12}{'RMSE':>12}")
    print("-" * 42)
    print(f"{'训练集':<8}{train_r2:>10.4f}{train_mse:>12.6f}{train_rmse:>12.6f}")
    print(f"{'验证集':<8}{val_r2:>10.4f}{val_mse:>12.6f}{val_rmse:>12.6f}")
    print(f"{'测试集':<8}{test_r2:>10.4f}{test_mse:>12.6f}{test_rmse:>12.6f}")

    print("\n" + "=" * 70)
    print("保存模型")
    print("=" * 70)
    model.save(save_path)


if __name__ == '__main__':
    main()
