import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from P1Q2 import MBGDNet, load_data, split_data
import json
import pickle


def evaluate_fingerprints(filepath='TLC_dataset.xlsx'):
    """
    任务1: 比较不同指纹的性能

    固定网络结构，仅比较不同的指纹类型、长度和半径设置
    """
    print("=" * 60)
    print("Task 1: Fingerprint Performance Comparison")
    print("=" * 60)

    # 定义要测试的指纹配置
    fp_configs = [
        # 不同指纹类型
        {'fp_type': 'morgan', 'fpSize': 128, 'radius': 3},
        {'fp_type': 'morgan', 'fpSize': 256, 'radius': 3},
        {'fp_type': 'morgan', 'fpSize': 128, 'radius': 2},
        {'fp_type': 'atompair', 'fpSize': 128},
        {'fp_type': 'rdkit', 'fpSize': 128},
        {'fp_type': 'topological', 'fpSize': 128},
    ]

    # 固定网络结构
    fixed_hidden_layers = [64, 32]
    print(f"\nFixed network architecture: {fixed_hidden_layers}")
    print("Output activation: sigmoid")
    print("=" * 60)

    results = []

    for config in fp_configs:
        print(f"\nTesting config: {config}")

        # 加载数据
        X, Y = load_data(filepath, **config)
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, Y)

        # 使用固定的网络结构
        layers_config = [
            (X.shape[1], fixed_hidden_layers[0], 'relu'),
            (fixed_hidden_layers[0], fixed_hidden_layers[1], 'relu'),
            (fixed_hidden_layers[1], 1, 'sigmoid')
        ]

        model = MBGDNet(layers_config, output_activation='sigmoid')

        # 训练
        train_losses, val_losses = model.train(
            X_train, y_train, X_val, y_val,
            epochs=500, batch_size=32, learning_rate=0.1, patience=30
        )

        # 评估
        _, r2, mse, rmse = model.evaluate(X_test, y_test)

        result = {
            'config': config,
            'fp_type': config['fp_type'],
            'fpSize': config.get('fpSize', '-'),
            'radius': config.get('radius', '-'),
            'input_dim': X.shape[1],
            'r2': float(r2),
            'mse': float(mse),
            'rmse': float(rmse),
            'final_val_loss': float(val_losses[-1])
        }
        results.append(result)

        print(f"R2: {r2:.4f}, MSE: {mse:.6f}, RMSE: {rmse:.6f}")

    # 保存结果
    with open('task1_fingerprint_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # 可视化结果 - x轴仅显示分子指纹类型
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 生成简化的指纹标签
    def get_fp_label(config):
        fp_type = config['fp_type']
        fp_size = config.get('fpSize', '')
        radius = config.get('radius', '')
        if fp_type == 'morgan':
            return f"Morgan\nr={radius}, s={fp_size}"
        else:
            return f"{fp_type.capitalize()}\ns={fp_size}"

    labels = [get_fp_label(r['config']) for r in results]
    r2_scores = [r['r2'] for r in results]
    mses = [r['mse'] for r in results]
    rmses = [r['rmse'] for r in results]

    axes[0].bar(range(len(results)), r2_scores, color='skyblue', edgecolor='black')
    axes[0].set_title('R2 Score by Fingerprint')
    axes[0].set_ylabel('R2 Score')
    axes[0].set_xticks(range(len(results)))
    axes[0].set_xticklabels(labels, rotation=0, ha='center', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(range(len(results)), mses, color='lightcoral', edgecolor='black')
    axes[1].set_title('MSE by Fingerprint')
    axes[1].set_ylabel('MSE')
    axes[1].set_xticks(range(len(results)))
    axes[1].set_xticklabels(labels, rotation=0, ha='center', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(range(len(results)), rmses, color='lightgreen', edgecolor='black')
    axes[2].set_title('RMSE by Fingerprint')
    axes[2].set_ylabel('RMSE')
    axes[2].set_xticks(range(len(results)))
    axes[2].set_xticklabels(labels, rotation=0, ha='center', fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('task1_fingerprint_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印最佳配置
    best_idx = np.argmax(r2_scores)
    print(f"\nBest fingerprint: {results[best_idx]['fp_type']}")
    print(f"  Size: {results[best_idx]['fpSize']}, Radius: {results[best_idx]['radius']}")
    print(f"  R2 Score: {results[best_idx]['r2']:.4f}")

    return results


def evaluate_output_activations(filepath='TLC_dataset.xlsx'):
    """
    任务2: 比较sigmoid和线性输出激活

    对于线性激活，不使用激活函数，模型会自动学习值在[0,1]范围内
    """
    print("\n" + "=" * 60)
    print("Task 2: Output Activation Comparison (Sigmoid vs Linear)")
    print("=" * 60)

    # 加载数据 (使用默认配置)
    np.random.seed(42)
    X, Y = load_data(filepath, fp_type='morgan', fpSize=128)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, Y)

    results = {}

    # 测试不同输出激活
    output_activations = ['sigmoid', 'linear']
    colors = {'sigmoid': 'blue', 'linear': 'red'}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for act in output_activations:
        print(f"\n{'='*40}")
        print(f"Testing output activation: {act}")
        print('='*40)

        layers_config = [
            (X.shape[1], 64, 'relu'),
            (64, 32, 'relu'),
            (32, 1, act)
        ]

        model = MBGDNet(layers_config, output_activation=act)

        # 训练
        train_losses, val_losses = model.train(
            X_train, y_train, X_val, y_val,
            epochs=500, batch_size=32, learning_rate=0.1, patience=30
        )

        # 评估
        y_pred, r2, mse, rmse = model.evaluate(X_test, y_test)

        results[act] = {
            'r2': float(r2),
            'mse': float(mse),
            'rmse': float(rmse),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'y_pred': y_pred
        }

        print(f"\n{act.upper()} Results:")
        print(f"  R2 Score: {r2:.4f}")
        print(f"  MSE: {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")

        # 检查预测值范围
        print(f"  Predicted range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")

        # 绘制
        row = 0 if act == 'sigmoid' else 1

        # Training history
        axes[row, 0].plot(train_losses, label=f'{act} Train', color=colors[act], linestyle='-')
        axes[row, 0].plot(val_losses, label=f'{act} Val', color=colors[act], linestyle='--')
        axes[row, 0].set_title(f'{act.capitalize()}: Training History')
        axes[row, 0].set_xlabel('Epoch')
        axes[row, 0].set_ylabel('MSE Loss')
        axes[row, 0].legend()
        axes[row, 0].grid(True, alpha=0.3)

        # Prediction vs Actual
        axes[row, 1].scatter(y_test, y_pred, alpha=0.5, color=colors[act],
                            edgecolors='black', linewidth=0.5)
        axes[row, 1].plot([0, 1], [0, 1], 'k--', label='Perfect')
        axes[row, 1].set_title(f'{act.capitalize()}: R2={r2:.4f}')
        axes[row, 1].set_xlabel('Actual Rf')
        axes[row, 1].set_ylabel('Predicted Rf')
        axes[row, 1].legend()
        axes[row, 1].grid(True, alpha=0.3)
        axes[row, 1].set_xlim(0, 1)
        axes[row, 1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('task2_activation_comparison.png', dpi=300)
    plt.show()

    # 对比总结
    print("\n" + "="*60)
    print("Summary: Sigmoid vs Linear Output Activation")
    print("="*60)
    print(f"{'Metric':<20} {'Sigmoid':<15} {'Linear':<15}")
    print("-" * 50)
    print(f"{'R2 Score':<20} {results['sigmoid']['r2']:<15.4f} {results['linear']['r2']:<15.4f}")
    print(f"{'MSE':<20} {results['sigmoid']['mse']:<15.6f} {results['linear']['mse']:<15.6f}")
    print(f"{'RMSE':<20} {results['sigmoid']['rmse']:<15.6f} {results['linear']['rmse']:<15.6f}")

    # 保存结果
    with open('task2_activation_results.json', 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk not in ['train_losses', 'val_losses', 'y_pred']}
                   for k, v in results.items()}, f, indent=2)

    return results


def evaluate_architectures(filepath='TLC_dataset.xlsx'):
    """
    任务3: 比较不同的模型结构

    比较神经网络的层数和节点数对性能的影响
    """
    print("\n" + "=" * 60)
    print("Task 3: Model Architecture Comparison")
    print("=" * 60)

    # 加载数据
    np.random.seed(42)
    X, Y = load_data(filepath, fp_type='morgan', fpSize=128)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, Y)

    # 定义不同的网络架构
    architectures = [
        # 浅层网络
        {'name': 'Shallow-Small', 'layers': [32]},
        {'name': 'Shallow-Medium', 'layers': [64]},
        {'name': 'Shallow-Large', 'layers': [128]},

        # 双层网络
        {'name': '2Layer-Small', 'layers': [32, 16]},
        {'name': '2Layer-Medium', 'layers': [64, 32]},
        {'name': '2Layer-Large', 'layers': [128, 64]},

        # 深层网络
        {'name': '3Layer-Small', 'layers': [64, 32, 16]},
        {'name': '3Layer-Medium', 'layers': [128, 64, 32]},
        {'name': '3Layer-Large', 'layers': [256, 128, 64]},

        # 特别深的网络
        {'name': '4Layer-Medium', 'layers': [64, 64, 32, 16]},
    ]

    results = []

    for arch in architectures:
        print(f"\n{'='*50}")
        print(f"Testing architecture: {arch['name']}")
        print(f"Hidden layers: {arch['layers']}")
        print('='*50)

        # 创建层配置
        layers_config = []
        prev_dim = X.shape[1]
        for hidden_dim in arch['layers']:
            layers_config.append((prev_dim, hidden_dim, 'relu'))
            prev_dim = hidden_dim
        layers_config.append((prev_dim, 1, 'sigmoid'))

        model = MBGDNet(layers_config, output_activation='sigmoid')
        param_count = model.get_params_count()

        # 训练
        train_losses, val_losses = model.train(
            X_train, y_train, X_val, y_val,
            epochs=500, batch_size=32, learning_rate=0.1, patience=30
        )

        # 评估
        y_pred, r2, mse, rmse = model.evaluate(X_test, y_test)

        result = {
            'name': arch['name'],
            'layers': arch['layers'],
            'param_count': param_count,
            'r2': float(r2),
            'mse': float(mse),
            'rmse': float(rmse),
            'final_train_loss': float(train_losses[-1]),
            'final_val_loss': float(val_losses[-1])
        }
        results.append(result)

        print(f"\nParameters: {param_count}")
        print(f"R2: {r2:.4f}, MSE: {mse:.6f}, RMSE: {rmse:.6f}")

    # 可视化结果
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    names = [r['name'] for r in results]
    r2_scores = [r['r2'] for r in results]
    param_counts = [r['param_count'] for r in results]
    rmses = [r['rmse'] for r in results]

    # R2 Score comparison
    axes[0, 0].bar(range(len(results)), r2_scores, color='steelblue', edgecolor='black')
    axes[0, 0].set_title('R2 Score by Architecture')
    axes[0, 0].set_ylabel('R2 Score')
    axes[0, 0].set_xticks(range(len(results)))
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # RMSE comparison
    axes[0, 1].bar(range(len(results)), rmses, color='coral', edgecolor='black')
    axes[0, 1].set_title('RMSE by Architecture')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_xticks(range(len(results)))
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Parameters vs Performance
    scatter = axes[1, 0].scatter(param_counts, r2_scores, c=rmses, cmap='RdYlGn_r',
                                 s=100, edgecolors='black', linewidth=1)
    axes[1, 0].set_xlabel('Number of Parameters')
    axes[1, 0].set_ylabel('R2 Score')
    axes[1, 0].set_title('Parameters vs R2 Score (color=RMSE)')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 0], label='RMSE')

    # Add labels for each point
    for i, name in enumerate(names):
        axes[1, 0].annotate(name, (param_counts[i], r2_scores[i]),
                           fontsize=6, ha='center')

    # Layer depth comparison
    depth_groups = {}
    for r in results:
        depth = len(r['layers'])
        if depth not in depth_groups:
            depth_groups[depth] = []
        depth_groups[depth].append(r['r2'])

    depths = sorted(depth_groups.keys())
    depth_means = [np.mean(depth_groups[d]) for d in depths]
    depth_stds = [np.std(depth_groups[d]) for d in depths]

    axes[1, 1].bar(depths, depth_means, yerr=depth_stds, capsize=5,
                   color='mediumpurple', edgecolor='black')
    axes[1, 1].set_xlabel('Number of Hidden Layers')
    axes[1, 1].set_ylabel('Average R2 Score')
    axes[1, 1].set_title('Performance vs Network Depth')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('task3_architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印最佳架构
    best_idx = np.argmax(r2_scores)
    print("\n" + "="*60)
    print("Best Architecture:")
    print(f"  Name: {results[best_idx]['name']}")
    print(f"  Layers: {results[best_idx]['layers']}")
    print(f"  Parameters: {results[best_idx]['param_count']}")
    print(f"  R2 Score: {results[best_idx]['r2']:.4f}")
    print(f"  RMSE: {results[best_idx]['rmse']:.6f}")
    print("="*60)

    # 保存结果
    with open('task3_architecture_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def train_and_save_best_model(filepath='TLC_dataset.xlsx'):
    """
    根据所有评估结果，训练效果最好的模型并保存为.pkl文件

    最佳模型配置基于之前三个任务的评估结果：
    - 最佳指纹类型：morgan, fpSize=256, radius=3
    - 最佳输出激活：sigmoid
    - 最佳架构：2Layer-Medium [64, 32]
    """
    print("\n" + "=" * 70)
    print("Training and Saving Best Model")
    print("=" * 70)

    # 加载最佳配置的数据
    np.random.seed(42)
    X, Y = load_data(filepath, fp_type='morgan', fpSize=256, radius=3)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, Y)

    print(f"\nData loaded:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Validation samples: {X_val.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Input features: {X_train.shape[1]}")

    # 创建最佳模型架构
    layers_config = [
        (X.shape[1], 64, 'relu'),  # 输入层 -> 64
        (64, 32, 'relu'),          # 64 -> 32
        (32, 1, 'sigmoid')         # 32 -> 输出 (sigmoid激活)
    ]

    model = MBGDNet(layers_config, output_activation='sigmoid')
    print(f"\nModel architecture:")
    print(f"  {layers_config}")
    print(f"  Total parameters: {model.get_params_count()}")

    # 训练模型
    print("\nTraining best model...")
    train_losses, val_losses = model.train(
        X_train, y_train, X_val, y_val,
        epochs=500, batch_size=32, learning_rate=0.1, patience=30
    )

    # 评估模型
    y_pred, r2, mse, rmse = model.evaluate(X_test, y_test)

    print(f"\n" + "=" * 70)
    print("Best Model Performance:")
    print("=" * 70)
    print(f"  R2 Score: {r2:.4f}")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  Predicted range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")

    # 保存模型为.pkl文件
    model_data = {
        'layers_config': layers_config,
        'output_activation': 'sigmoid',
        'weights': model.weights,
        'biases': model.biases,
        'performance': {
            'r2': r2,
            'mse': mse,
            'rmse': rmse
        },
        'training_info': {
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1]
        }
    }

    with open('best_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\nModel saved to: best_model.pkl")

    # 可视化训练过程和预测结果
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 训练历史
    axes[0].plot(train_losses, label='Train Loss', color='blue')
    axes[0].plot(val_losses, label='Val Loss', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Best Model Training History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 预测 vs 实际
    axes[1].scatter(y_test, y_pred, alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[1].plot([0, 1], [0, 1], 'r--', label=f'Perfect (R2={r2:.4f})')
    axes[1].set_xlabel('Actual Rf')
    axes[1].set_ylabel('Predicted Rf')
    axes[1].set_title('Best Model: Prediction vs Actual')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('best_model_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Performance plot saved to: best_model_performance.png")
    print("=" * 70)

    return model, model_data


def load_and_test_model(filepath='TLC_dataset.xlsx'):
    """
    加载保存的模型并测试
    """
    print("\n" + "=" * 70)
    print("Loading and Testing Saved Model")
    print("=" * 70)

    # 加载模型
    with open('best_model.pkl', 'rb') as f:
        model_data = pickle.load(f)

    print(f"\nLoaded model configuration:")
    print(f"  Layers: {model_data['layers_config']}")
    print(f"  Output activation: {model_data['output_activation']}")
    print(f"  Saved R2: {model_data['performance']['r2']:.4f}")

    # 重新创建模型
    model = MBGDNet(model_data['layers_config'], output_activation=model_data['output_activation'])
    model.weights = model_data['weights']
    model.biases = model_data['biases']

    # 加载数据并测试
    X, Y = load_data(filepath, fp_type='morgan', fpSize=256, radius=3)
    _, _, X_test, y_test, _, _ = split_data(X, Y)

    y_pred, r2, mse, rmse = model.evaluate(X_test, y_test)

    print(f"\nReloaded model performance:")
    print(f"  R2 Score: {r2:.4f}")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print("=" * 70)

    return model


def run_all_evaluations():
    """
    运行所有三个评估任务，并训练保存最佳模型
    """
    print("\n" + "="*70)
    print("Running All Evaluation Tasks")
    print("="*70)

    # Task 1: 指纹比较
    fp_results = evaluate_fingerprints()

    # Task 2: 输出激活函数比较
    act_results = evaluate_output_activations()

    # Task 3: 模型架构比较
    arch_results = evaluate_architectures()

    # 训练并保存最佳模型
    best_model, model_data = train_and_save_best_model()

    # 测试加载保存的模型
    loaded_model = load_and_test_model()

    print("\n" + "="*70)
    print("All evaluations completed!")
    print("Results saved to:")
    print("  - task1_fingerprint_results.json")
    print("  - task1_fingerprint_comparison.png")
    print("  - task2_activation_results.json")
    print("  - task2_activation_comparison.png")
    print("  - task3_architecture_results.json")
    print("  - task3_architecture_comparison.png")
    print("  - best_model.pkl")
    print("  - best_model_performance.png")
    print("="*70)

    return fp_results, act_results, arch_results, best_model, model_data


if __name__ == "__main__":
    run_all_evaluations()
