import pubchem_compounds as pc
import pickle
import numpy as np
import matplotlib.pyplot as plt
from P1Q2 import MBGDNet, smiles_to_fingerprint


def cas_to_smiles(cas_number: str) -> str | None:
    """
    将CAS号转换为SMILES字符串。

    使用pubchem-compounds库进行查询。

    Args:
        cas_number: CAS号，例如 "50-00-0" (甲醛)

    Returns:
        SMILES字符串，如果未找到则返回None
    """
    cas_number = cas_number.strip()

    try:
        results, failed = pc.cas_to_smiles(cas_number)
        if cas_number in results:
            return results[cas_number]
        return None
    except Exception as e:
        print(f"解析错误: {e}")
        return None


def load_model(model_path='best_model.pkl'):
    """
    加载训练好的神经网络模型。

    Args:
        model_path: 模型文件路径

    Returns:
        加载好的MBGDNet模型实例
    """
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    # 重新创建模型
    model = MBGDNet(model_data['layers_config'],
                    output_activation=model_data['output_activation'])
    model.weights = model_data['weights']
    model.biases = model_data['biases']

    return model


def predict_rf(model, fingerprint, solvent_composition):
    """
    使用模型预测化合物的Rf值。

    Args:
        model: 神经网络模型
        fingerprint: 化合物指纹向量 (numpy array)
        solvent_composition: 洗脱液组成 [H, EA, DCM, MeOH, Et2O]

    Returns:
        预测的Rf值
    """
    # 拼接指纹和洗脱液组成
    x = np.concatenate([fingerprint, solvent_composition]).reshape(1, -1)
    rf = model.predict(x)[0]
    return rf


def find_optimal_solvent(
    target_fp,
    other_fp,
    model_path='best_model.pkl',
    search_method='grid',
    num_samples=10000
):
    """
    找到合适的洗脱液组成，使得目标化合物的Rf值在0.2~0.3之间，
    并且其他化合物与目标化合物的Rf值差大于0.1。

    Args:
        target_fp: 目标化合物的指纹 (numpy array)，形状为 (fp_size,)
        other_fp: 其他化合物的指纹 (numpy array)，形状为 (n_compounds-1, fp_size) 或 (fp_size,)
        target_idx: 目标化合物在other_fp中的索引（如果other_fp是二维数组）
        model_path: 模型文件路径
        search_method: 搜索方法，'grid'为网格搜索，'random'为随机搜索
        num_samples: 随机搜索时的采样数量

    Returns:
        dict: 包含最佳洗脱液组成和预测结果的字典
            {
                'solvent': [H, EA, DCM, MeOH, Et2O],
                'target_rf': 目标化合物Rf值,
                'other_rf': 其他化合物Rf值列表,
                'rf_differences': Rf差值列表,
                'min_difference': 最小Rf差值,
                'target_in_range': 目标Rf是否在0.2~0.3范围内
            }
        如果未找到满足条件的解，返回None
    """
    # 加载模型
    model = load_model(model_path)

    # 确保指纹是numpy数组
    target_fp = np.array(target_fp).flatten()

    # 处理other_fp - 可能是单个指纹或多个指纹
    other_fps = []
    if len(other_fp.shape) == 1:
        # 单个指纹
        other_fps = [other_fp.flatten()]
    else:
        # 多个指纹
        other_fps = [fp.flatten() for fp in other_fp]

    # 定义洗脱液组成搜索空间
    # 根据数据集，常见的溶剂系统是：
    # 1. H + EA 系统 (DCM=0, MeOH=0, Et2O=0)
    # 2. DCM + MeOH 系统 (H=0, EA=0, Et2O=0)
    # 3. H + Et2O 系统 (EA=0, DCM=0, MeOH=0)

    best_result = None
    best_score = -float('inf')

    # 定义 solvent systems - 使用归一化比例
    solvent_systems = [
        # 二元混合系统
        {'name': 'H_EA', 'H': (0.0, 1.0), 'EA': (0.0, 1.0), 'DCM': 0, 'MeOH': 0, 'Et2O': 0},
        {'name': 'DCM_MeOH', 'H': 0, 'EA': 0, 'DCM': (0.0, 1.0), 'MeOH': (0.0, 1.0), 'Et2O': 0},
        {'name': 'H_Et2O', 'H': (0.0, 1.0), 'EA': 0, 'DCM': 0, 'MeOH': 0, 'Et2O': (0.0, 1.0)},
        # 纯溶剂（边界情况）
        {'name': 'pure_H', 'H': 1.0, 'EA': 0, 'DCM': 0, 'MeOH': 0, 'Et2O': 0},
        {'name': 'pure_EA', 'H': 0, 'EA': 1.0, 'DCM': 0, 'MeOH': 0, 'Et2O': 0},
        {'name': 'pure_DCM', 'H': 0, 'EA': 0, 'DCM': 1.0, 'MeOH': 0, 'Et2O': 0},
        {'name': 'pure_MeOH', 'H': 0, 'EA': 0, 'DCM': 0, 'MeOH': 1.0, 'Et2O': 0},
        {'name': 'pure_Et2O', 'H': 0, 'EA': 0, 'DCM': 0, 'MeOH': 0, 'Et2O': 1.0},
    ]

    def evaluate_solvent(solvent_comp):
        """评估一组洗脱液组成"""
        # 预测目标化合物Rf
        target_rf = predict_rf(model, target_fp, solvent_comp)

        # 预测其他化合物Rf
        other_rfs = [predict_rf(model, fp, solvent_comp) for fp in other_fps]

        # 计算与目标化合物的Rf差值
        rf_differences = [abs(target_rf - rf) for rf in other_rfs]
        min_diff = min(rf_differences) if rf_differences else 0

        # 检查条件
        target_in_range = 0.2 <= target_rf <= 0.3
        separation_ok = min_diff > 0.1

        # 评分函数：满足范围条件后，差值越高评分越高
        score = 0
        if target_in_range:
            score += 1000000  # 满足范围基础分

        if separation_ok:
            score += 50  # 分离条件满足加分
            score += min_diff * 100  # 差值越大越好（加权更高）

        return {
            'solvent': solvent_comp,
            'target_rf': target_rf,
            'other_rf': other_rfs,
            'rf_differences': rf_differences,
            'min_difference': min_diff,
            'target_in_range': target_in_range,
            'separation_ok': separation_ok,
            'score': score
        }

    if search_method == 'grid':
        # 网格搜索 - 使用归一化比例（与训练数据一致）
        steps = 20  # 0.0 到 1.0，步长0.05

        for system in solvent_systems:
            # 获取该溶剂系统的可变参数范围
            vars_to_search = []
            fixed_values = {}

            for solvent_name in ['H', 'EA', 'DCM', 'MeOH', 'Et2O']:
                val = system[solvent_name]
                if isinstance(val, tuple):
                    vars_to_search.append((solvent_name, val[0], val[1]))
                else:
                    fixed_values[solvent_name] = val

            # 处理纯溶剂情况（无变量）
            if len(vars_to_search) == 0:
                # 纯溶剂，直接评估
                solvent_comp = [
                    fixed_values.get('H', 0),
                    fixed_values.get('EA', 0),
                    fixed_values.get('DCM', 0),
                    fixed_values.get('MeOH', 0),
                    fixed_values.get('Et2O', 0)
                ]
                result = evaluate_solvent(solvent_comp)
                if result['score'] > best_score:
                    best_score = result['score']
                    best_result = result

            # 生成网格（二元混合系统）
            elif len(vars_to_search) == 2:
                name1, min1, max1 = vars_to_search[0]
                name2, min2, max2 = vars_to_search[1]

                for i in range(steps + 1):
                    for j in range(steps + 1):
                        # 生成比例值（0.0到1.0）
                        v1 = min1 + (max1 - min1) * i / steps
                        v2 = min2 + (max2 - min2) * j / steps

                        # 归一化使总和为1.0
                        total = v1 + v2
                        if total > 0:
                            v1_norm = v1 / total
                            v2_norm = v2 / total
                        else:
                            v1_norm = v2_norm = 0

                        solvent_comp = [
                            fixed_values.get('H', v1_norm if name1 == 'H' else v2_norm if name2 == 'H' else 0),
                            fixed_values.get('EA', v1_norm if name1 == 'EA' else v2_norm if name2 == 'EA' else 0),
                            fixed_values.get('DCM', v1_norm if name1 == 'DCM' else v2_norm if name2 == 'DCM' else 0),
                            fixed_values.get('MeOH', v1_norm if name1 == 'MeOH' else v2_norm if name2 == 'MeOH' else 0),
                            fixed_values.get('Et2O', v1_norm if name1 == 'Et2O' else v2_norm if name2 == 'Et2O' else 0)
                        ]

                        result = evaluate_solvent(solvent_comp)

                        if result['score'] > best_score:
                            best_score = result['score']
                            best_result = result

    else:
        # 随机搜索 - 使用归一化比例
        for system in solvent_systems:
            vars_to_search = []
            fixed_values = {}

            for solvent_name in ['H', 'EA', 'DCM', 'MeOH', 'Et2O']:
                val = system[solvent_name]
                if isinstance(val, tuple):
                    vars_to_search.append((solvent_name, val[0], val[1]))
                else:
                    fixed_values[solvent_name] = val

            samples_per_system = num_samples // len(solvent_systems)

            # 处理纯溶剂情况（无变量）
            if len(vars_to_search) == 0:
                solvent_comp = [
                    fixed_values.get('H', 0),
                    fixed_values.get('EA', 0),
                    fixed_values.get('DCM', 0),
                    fixed_values.get('MeOH', 0),
                    fixed_values.get('Et2O', 0)
                ]
                result = evaluate_solvent(solvent_comp)
                if result['score'] > best_score:
                    best_score = result['score']
                    best_result = result

            # 二元混合系统
            elif len(vars_to_search) == 2:
                for _ in range(samples_per_system):
                    # 随机生成比例并归一化
                    v1 = np.random.uniform(0, 1)
                    v2 = np.random.uniform(0, 1)
                    total = v1 + v2
                    if total > 0:
                        v1_norm = v1 / total
                        v2_norm = v2 / total
                    else:
                        v1_norm = v2_norm = 0.5

                    solvent_comp = [
                        fixed_values.get('H', v1_norm if vars_to_search[0][0] == 'H' else v2_norm if vars_to_search[1][0] == 'H' else 0),
                        fixed_values.get('EA', v1_norm if vars_to_search[0][0] == 'EA' else v2_norm if vars_to_search[1][0] == 'EA' else 0),
                        fixed_values.get('DCM', v1_norm if vars_to_search[0][0] == 'DCM' else v2_norm if vars_to_search[1][0] == 'DCM' else 0),
                        fixed_values.get('MeOH', v1_norm if vars_to_search[0][0] == 'MeOH' else v2_norm if vars_to_search[1][0] == 'MeOH' else 0),
                        fixed_values.get('Et2O', v1_norm if vars_to_search[0][0] == 'Et2O' else v2_norm if vars_to_search[1][0] == 'Et2O' else 0)
                    ]

                    result = evaluate_solvent(solvent_comp)

                    if result['score'] > best_score:
                        best_score = result['score']
                        best_result = result

    # 清理结果，移除评分相关的内部字段
    if best_result:
        result = {
            'solvent': {
                'H': best_result['solvent'][0],
                'EA': best_result['solvent'][1],
                'DCM': best_result['solvent'][2],
                'MeOH': best_result['solvent'][3],
                'Et2O': best_result['solvent'][4]
            },
            'target_rf': float(best_result['target_rf']),
            'other_rf': [float(rf) for rf in best_result['other_rf']],
            'rf_differences': [float(diff) for diff in best_result['rf_differences']],
            'min_difference': float(best_result['min_difference']),
            'target_in_range': best_result['target_in_range'],
            'separation_ok': best_result['separation_ok']
        }
        return result

    return None


def get_fingerprint_from_smiles(smiles, fp_type='morgan', radius=3, fpSize=256):
    """
    从SMILES字符串获取化合物指纹。

    Args:
        smiles: SMILES字符串
        fp_type: 指纹类型，默认'morgan'
        radius: Morgan指纹半径，默认3
        fpSize: 指纹长度，默认256（与最佳模型一致）

    Returns:
        指纹向量 (numpy array)
    """
    return smiles_to_fingerprint(smiles, fp_type=fp_type, radius=radius, fpSize=fpSize)


def calculate_fingerprint_similarity(fp1, fp2, method='tanimoto'):
    """
    计算两个分子指纹的相似度。

    Args:
        fp1: 指纹1 (numpy array)
        fp2: 指纹2 (numpy array)
        method: 相似度计算方法，'tanimoto'(默认) 或 'cosine'

    Returns:
        float: 相似度值 (0.0 ~ 1.0)
    """
    if method == 'tanimoto':
        # Tanimoto相似度 (Jaccard系数) - 适用于二进制指纹
        fp1_binary = (fp1 > 0).astype(int)
        fp2_binary = (fp2 > 0).astype(int)

        intersection = np.sum(fp1_binary & fp2_binary)
        union = np.sum(fp1_binary | fp2_binary)

        if union == 0:
            return 0.0
        return intersection / union

    elif method == 'cosine':
        # 余弦相似度
        norm1 = np.linalg.norm(fp1)
        norm2 = np.linalg.norm(fp2)

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(fp1, fp2) / (norm1 * norm2)

    else:
        raise ValueError(f"不支持的相似度方法: {method}")


def analyze_compound_similarity(query_smiles_list, data_path='TLC_dataset.xlsx',
                                 fp_config=None, top_k=10, method='tanimoto'):
    """
    分析给定化合物与训练集/测试集中所有化合物的分子指纹相似度。

    Args:
        query_smiles_list: 待查询化合物的SMILES列表
        data_path: 数据集文件路径
        fp_config: 指纹配置字典
        top_k: 返回相似度最高的前k个结果
        method: 相似度计算方法

    Returns:
        dict: 每个查询化合物的相似度分析结果
    """
    import pandas as pd

    if fp_config is None:
        fp_config = {'fp_type': 'morgan', 'fpSize': 256, 'radius': 3}

    # 确保输入是列表
    if isinstance(query_smiles_list, str):
        query_smiles_list = [query_smiles_list]

    # 加载数据集
    df = pd.read_excel(data_path)
    unique_compounds = df[['COMPOUND_ID', 'COMPOUND_SMILES', 'COMPOUND_ENG_NAME']].drop_duplicates()

    print("=" * 70)
    print(f"Fingerprint Similarity Analysis ({method.upper()})")
    print("=" * 70)
    print(f"Dataset: {len(unique_compounds)} unique compounds")
    print(f"Query compounds: {len(query_smiles_list)}")
    print("=" * 70)

    results = {}

    for idx, query_smiles in enumerate(query_smiles_list):
        print(f"\nQuery {idx+1}: {query_smiles[:50]}...")

        # 计算查询化合物的指纹
        try:
            query_fp = get_fingerprint_from_smiles(query_smiles, **fp_config)
        except Exception as e:
            print(f"Error generating fingerprint: {e}")
            continue

        # 计算与数据集中每个化合物的相似度
        similarities = []

        for _, row in unique_compounds.iterrows():
            dataset_smiles = row['COMPOUND_SMILES']

            if dataset_smiles == query_smiles:
                continue

            try:
                dataset_fp = get_fingerprint_from_smiles(dataset_smiles, **fp_config)
                similarity = calculate_fingerprint_similarity(query_fp, dataset_fp, method=method)

                similarities.append({
                    'compound_id': row['COMPOUND_ID'],
                    'smiles': dataset_smiles,
                    'name': row['COMPOUND_ENG_NAME'],
                    'similarity': similarity
                })
            except Exception:
                continue

        # 按相似度排序
        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        # 统计相似度分布
        high_sim = [s for s in similarities if s['similarity'] >= 0.9]
        med_sim = [s for s in similarities if 0.7 <= s['similarity'] < 0.9]
        low_sim = [s for s in similarities if s['similarity'] < 0.7]

        print(f"\nSimilarity Distribution:")
        print(f"  High (>=0.9): {len(high_sim)} compounds")
        print(f"  Medium (0.7-0.9): {len(med_sim)} compounds")
        print(f"  Low (<0.7): {len(low_sim)} compounds")

        # 显示最相似的前k个
        print(f"\nTop {top_k} Most Similar Compounds:")
        print("-" * 70)
        print(f"{'Rank':<6} {'ID':<8} {'Similarity':<12} {'Name':<40}")
        print("-" * 70)

        for i, sim in enumerate(similarities[:top_k], 1):
            name = sim['name'] if isinstance(sim['name'], str) else f"Compound_{sim['compound_id']}"
            name = name[:37] + '...' if len(name) > 40 else name
            print(f"{i:<6} {sim['compound_id']:<8} {sim['similarity']:<12.4f} {name:<40}")

        # 保存结果
        results[query_smiles] = {
            'fingerprint': query_fp,
            'all_similarities': similarities,
            'top_k': similarities[:top_k],
            'stats': {
                'high_similarity': len(high_sim),
                'medium_similarity': len(med_sim),
                'low_similarity': len(low_sim),
                'max_similarity': similarities[0]['similarity'] if similarities else 0,
                'avg_similarity': np.mean([s['similarity'] for s in similarities]) if similarities else 0
            }
        }

    return results


def analyze_functional_groups(smiles, name=""):
    """
    分析化合物中的官能团。

    Args:
        smiles: SMILES字符串
        name: 化合物名称（用于输出）

    Returns:
        dict: 包含检测到的官能团信息
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    functional_groups = {
        'halogen': False,
        'chlorine': False,
        'bromine': False,
        'fluorine': False,
        'iodine': False,
        'ether': False,
        'amine': False,
        'aromatic_ring': False,
        'phenyl': False,
        'biphenyl': False,
        'methyl': False,
        'hydroxyl': False,
        'carbonyl': False,
        'nitro': False,
        'methoxy': False,
        'morpholine': False,
        'toluene': False,  # 甲苯基
    }

    # SMARTS模式匹配
    patterns = {
        'halogen': '[F,Cl,Br,I]',
        'chlorine': '[Cl]',
        'bromine': '[Br]',
        'fluorine': '[F]',
        'iodine': '[I]',
        'ether': '[OX2]([#6])[#6]',
        'amine': '[NX3]',
        'aromatic_ring': 'c1ccccc1',
        'phenyl': 'c1ccccc1',
        'biphenyl': 'c1ccccc1-c2ccccc2',
        'methyl': '[CH3]',
        'hydroxyl': '[OH]',
        'carbonyl': '[#6]=[OX1]',
        'nitro': '[N+](=O)[O-]',
        'methoxy': 'COc',
        'morpholine': 'C1COCCN1',
        'toluene': 'Cc1ccccc1',
    }

    for group, smarts in patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            functional_groups[group] = True

    # 计算分子描述符
    functional_groups['mol_weight'] = rdMolDescriptors.CalcExactMolWt(mol)
    functional_groups['num_atoms'] = mol.GetNumAtoms()
    functional_groups['num_aromatic_rings'] = rdMolDescriptors.CalcNumAromaticRings(mol)

    return functional_groups


def check_functional_group_coverage(query_compounds, data_path='TLC_dataset.xlsx'):
    """
    检查查询化合物中的官能团在数据集中的覆盖情况。

    Args:
        query_compounds: 查询化合物字典 {name: smiles}
        data_path: 数据集路径

    Returns:
        dict: 官能团覆盖分析结果
    """
    import pandas as pd

    print("\n" + "=" * 70)
    print("Functional Group Coverage Analysis")
    print("=" * 70)

    # 加载数据集
    df = pd.read_excel(data_path)
    unique_smiles = df['COMPOUND_SMILES'].unique()

    print(f"Dataset size: {len(unique_smiles)} unique compounds")
    print(f"Query compounds: {len(query_compounds)}")

    # 分析查询化合物的官能团
    print("\n" + "=" * 70)
    print("Query Compounds Functional Groups")
    print("=" * 70)

    query_fg_summary = {}
    all_query_groups = set()

    for name, smiles in query_compounds.items():
        fg = analyze_functional_groups(smiles, name)
        query_fg_summary[name] = fg

        print(f"\n{name}:")
        print(f"  SMILES: {smiles}")
        print(f"  Mol. Weight: {fg['mol_weight']:.2f}")
        print(f"  Aromatic rings: {fg['num_aromatic_rings']}")
        print(f"  Functional groups: ", end='')

        detected = [g for g, v in fg.items() if v is True]
        print(', '.join(detected) if detected else 'None')

        all_query_groups.update(detected)

    # 分析数据集中所有化合物的官能团
    print("\n" + "=" * 70)
    print("Checking Dataset Coverage...")
    print("=" * 70)

    dataset_fg_summary = {
        'halogen': [],
        'chlorine': [],
        'ether': [],
        'aromatic_ring': [],
        'phenyl': [],
        'biphenyl': [],
        'methyl': [],
        'methoxy': [],
        'morpholine': [],
        'toluene': [],
    }

    print(f"\nAnalyzing {len(unique_smiles)} compounds in dataset...")

    for i, smiles in enumerate(unique_smiles):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(unique_smiles)} compounds...")

        try:
            fg = analyze_functional_groups(smiles)
            if fg:
                for group in dataset_fg_summary.keys():
                    if fg.get(group, False):
                        dataset_fg_summary[group].append(smiles)
        except Exception:
            continue

    # 输出覆盖情况
    print("\n" + "=" * 70)
    print("Functional Group Coverage Summary")
    print("=" * 70)
    print(f"{'Functional Group':<25} {'Query Has':<12} {'Dataset Count':<15} {'Coverage':<12}")
    print("-" * 70)

    coverage_results = {}

    for group in sorted(all_query_groups):
        if group in ['mol_weight', 'num_atoms', 'num_aromatic_rings']:
            continue

        query_has = any(query_fg_summary[name].get(group, False) for name in query_compounds)
        dataset_count = len(dataset_fg_summary.get(group, []))
        coverage = "✅" if dataset_count > 0 else "❌"

        print(f"{group:<25} {'Yes' if query_has else 'No':<12} {dataset_count:<15} {coverage:<12}")

        coverage_results[group] = {
            'query_has': query_has,
            'dataset_count': dataset_count,
            'covered': dataset_count > 0
        }

    # 检查缺失的官能团
    missing_groups = [g for g, r in coverage_results.items() if not r['covered'] and r['query_has']]

    print("\n" + "=" * 70)
    if missing_groups:
        print("⚠️  WARNING: Missing functional groups in dataset:")
        for g in missing_groups:
            print(f"  - {g}")
        print("\nThese groups exist in query compounds but NOT in training data!")
        print("This may cause poor model prediction performance.")
    else:
        print("✅ All functional groups in query compounds are covered in dataset!")

    return {
        'query_fg': query_fg_summary,
        'dataset_fg_counts': {k: len(v) for k, v in dataset_fg_summary.items()},
        'coverage': coverage_results,
        'missing_in_dataset': missing_groups
    }


def test_solvent_optimization(smiles1="CCO", smiles2="c1ccccc1",
                               model_path='best_model.pkl', verbose=True):
    """
    洗脱液优化功能。

    使用两个化合物的SMILES，查找最佳洗脱液组成，
    使得化合物1的Rf在0.2~0.3之间，且与化合物2的Rf差>0.1。

    Args:
        smiles1: 目标化合物SMILES
        smiles2: 其他化合物SMILES
        model_path: 模型文件路径
        verbose: 是否打印详细信息

    Returns:
        dict: 优化结果，包含最佳洗脱液组成和预测Rf值
              如果失败返回None
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Test: Solvent optimization")
        print("=" * 60)

    try:
        # 获取化合物指纹
        fp1 = get_fingerprint_from_smiles(smiles1)
        fp2 = get_fingerprint_from_smiles(smiles2)

        if verbose:
            print(f"\nCompound 1 fingerprint shape: {fp1.shape}")
            print(f"Compound 2 fingerprint shape: {fp2.shape}")

        # 查找最佳洗脱液组成
        result = find_optimal_solvent(
            target_fp=fp1,
            other_fp=fp2,
            search_method='grid'
        )

        if result and verbose:
            print("\n" + "=" * 60)
            print("Optimal solvent composition found!")
            print("=" * 60)
            # 转换为百分比显示（总和=100%）
            h_pct = result['solvent']['H'] * 100
            ea_pct = result['solvent']['EA'] * 100
            dcm_pct = result['solvent']['DCM'] * 100
            meoh_pct = result['solvent']['MeOH'] * 100
            et2o_pct = result['solvent']['Et2O'] * 100

            print(f"Solvent composition (normalized to 100%):")
            if h_pct > 0:
                print(f"  H (正己烷): {h_pct:.1f}%")
            if ea_pct > 0:
                print(f"  EA (乙酸乙酯): {ea_pct:.1f}%")
            if dcm_pct > 0:
                print(f"  DCM (二氯甲烷): {dcm_pct:.1f}%")
            if meoh_pct > 0:
                print(f"  MeOH (甲醇): {meoh_pct:.1f}%")
            if et2o_pct > 0:
                print(f"  Et2O (乙醚): {et2o_pct:.1f}%")

            # 验证总和
            total_pct = h_pct + ea_pct + dcm_pct + meoh_pct + et2o_pct
            print(f"\nTotal: {total_pct:.1f}%")

            print(f"\nPredicted Rf values:")
            print(f"  Target compound: {result['target_rf']:.4f}")
            print(f"  Other compound: {result['other_rf'][0]:.4f}")
            print(f"  Rf difference: {result['rf_differences'][0]:.4f}")
            print(f"\nCondition check:")
            print(f"  Target Rf in [0.2, 0.3]: {result['target_in_range']}")
            print(f"  Separation > 0.1: {result['separation_ok']}")
        elif not result and verbose:
            print("No optimal solvent composition found!")

        return result

    except FileNotFoundError:
        if verbose:
            print(f"Model file '{model_path}' not found. Please train the model first.")
        return None
    except Exception as e:
        if verbose:
            print(f"Error during optimization: {e}")
            import traceback
            traceback.print_exc()
        return None


# 测试示例
if __name__ == "__main__":
    """
    # 测试CAS转SMILES
    # 甲醛
    cas = "50-00-0"
    smiles = cas_to_smiles(cas)
    print(f"CAS: {cas} -> SMILES: {smiles}")

    # 乙醇
    cas = "64-17-5"
    smiles = cas_to_smiles(cas)
    print(f"CAS: {cas} -> SMILES: {smiles}")

    # 苯
    cas = "71-43-2"
    smiles = cas_to_smiles(cas)
    print(f"CAS: {cas} -> SMILES: {smiles}")
    """
    # 绘制3个分子的实验和预测Rf值
    fig, axes = plot_compound_predictions(
        compound_ids=[1, 3, 4],
        model_path='best_model.pkl',
        data_path='TLC_dataset.xlsx'
    )
    plt.show()
    # 测试洗脱液优化
    #对甲苯氯,前一个1代表反应1，后一个1代表反应物1，下同
    cas_1_1 = "106-43-4"
    smiles_1_1 = cas_to_smiles(cas_1_1)
    #4-(对甲苯基)吗啉
    cas_1_2 = "3077-16-5"
    smiles_1_2 = cas_to_smiles(cas_1_2)

    result1 = test_solvent_optimization(
        smiles1=smiles_1_2,       
        smiles2=smiles_1_1   
    )

    #3-氯苯甲醚
    cas_2_3 = "2845-89-8"
    smiles_2_3 = cas_to_smiles(cas_2_3)
    #3-甲氧基联苯
    cas_2_4 = "2113-56-6"
    smiles_2_4 = cas_to_smiles(cas_2_4)
    result2 = test_solvent_optimization(
        smiles1=smiles_2_4,
        smiles2=smiles_2_3
    )

    # 分析四个化合物与训练集的指纹相似度
    print("\n" + "=" * 60)
    print("Fingerprint Similarity Analysis for 4 Compounds")
    print("=" * 60)

    # 四个待分析的化合物
    four_compounds = {
        'cas_1_1': smiles_1_1,  # 对甲苯氯
        'cas_1_2': smiles_1_2,  # 4-(对甲苯基)吗啉
        'cas_2_3': smiles_2_3,  # 3-氯苯甲醚
        'cas_2_4': smiles_2_4,  # 3-甲氧基联苯
    }

    # 转换为列表进行分析
    smiles_list = list(four_compounds.values())

    # 执行相似度分析
    similarity_results = analyze_compound_similarity(
        smiles_list,
        data_path='TLC_dataset.xlsx',
        top_k=10,
        method='tanimoto'
    )

    # 打印四个化合物之间的相互相似度
    print("\n" + "=" * 70)
    print("Inter-Compound Similarity Matrix")
    print("=" * 70)

    compound_names = ['对甲苯氯', '4-(对甲苯基)吗啉', '3-氯苯甲醚', '3-甲氧基联苯']

    # 打印表头
    print(f"{'Compound':<25}", end='')
    for name in compound_names:
        print(f"{name[:10]:<12}", end='')
    print()
    print("-" * 70)

    # 打印相似度矩阵
    smiles_keys = list(four_compounds.keys())
    for i, (key1, smiles1) in enumerate(four_compounds.items()):
        print(f"{compound_names[i]:<25}", end='')

        fp1 = get_fingerprint_from_smiles(smiles1)

        for j, (key2, smiles2) in enumerate(four_compounds.items()):
            if i == j:
                print(f"{'1.0000':<12}", end='')
            else:
                fp2 = get_fingerprint_from_smiles(smiles2)
                sim = calculate_fingerprint_similarity(fp1, fp2, method='tanimoto')
                print(f"{sim:<12.4f}", end='')
        print()

    # 检查4个化合物的官能团在数据集中的覆盖情况
    print("\n" + "=" * 70)
    print("检查4个化合物的官能团覆盖情况")
    print("=" * 70)

    four_compounds_dict = {
        '对甲苯氯': smiles_1_1,
        '4-(对甲苯基)吗啉': smiles_1_2,
        '3-氯苯甲醚': smiles_2_3,
        '3-甲氧基联苯': smiles_2_4,
    }

    fg_coverage = check_functional_group_coverage(
        four_compounds_dict,
        data_path='TLC_dataset.xlsx'
    )

