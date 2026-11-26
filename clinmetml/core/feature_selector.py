"""
Feature Selection Module for Auto Pipeline
Converts R-based sampling and matching logic to Python
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Set
from collections import Counter, defaultdict
import glob

# ClinMetML path management
from ..utils.paths import (
    get_feature_selection_dir, 
    get_feature_selection_subdir
)

# 可选依赖
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: lightgbm not available. LightGBM method will be disabled.")

try:
    import pymrmr
    PYMRMR_AVAILABLE = True
except ImportError:
    PYMRMR_AVAILABLE = False
    print("Warning: pymrmr not available. mRMR method will be disabled.")

try:
    from skrebate import ReliefF
    RELIEF_AVAILABLE = True
except ImportError:
    RELIEF_AVAILABLE = False
    print("Warning: skrebate not available. ReliefF method will be disabled.")

try:
    from skfeature.function.information_theoretical_based import FCBF
    FCBF_AVAILABLE = True
    FCBF_TYPE = 'skfeature'
except ImportError:
    try:
        from fcbf import fcbf
        FCBF_AVAILABLE = True
        FCBF_TYPE = 'fcbf'
    except ImportError:
        try:
            from FCBF_module import FCBF
            FCBF_AVAILABLE = True
            FCBF_TYPE = 'FCBF_module'
        except ImportError:
            FCBF_AVAILABLE = False
            FCBF_TYPE = None
            print("Warning: FCBF module not available. FCBF method will be disabled.")


def analyze_dataset_balance(data_path: str, target_col: str, balance_threshold: float = 0.3) -> Dict[str, any]:
    """
    分析数据集的平衡性
    
    Args:
        data_path: 数据文件路径
        target_col: 目标列名
        balance_threshold: 平衡阈值，如果少数类占比小于此值则认为不平衡
        
    Returns:
        Dict: 包含数据集分析结果的字典
    """
    print("🔍 分析数据集平衡性...")
    print("=" * 50)
    
    try:
        # 读取数据
        df = pd.read_csv(data_path)
        
        if target_col not in df.columns:
            raise ValueError(f"目标列 '{target_col}' 不存在于数据中")
        
        # 统计目标变量分布
        target_counts = df[target_col].value_counts().sort_index()
        total_samples = len(df)
        
        print(f"📊 数据集基本信息:")
        print(f"   总样本数: {total_samples}")
        print(f"   特征数: {len(df.columns) - 1}")
        print(f"   目标列: {target_col}")
        
        print(f"\n📈 目标变量分布:")
        for value, count in target_counts.items():
            percentage = (count / total_samples) * 100
            print(f"   类别 {value}: {count} 样本 ({percentage:.1f}%)")
        
        # 判断数据集平衡性
        min_class_ratio = target_counts.min() / total_samples
        is_balanced = min_class_ratio >= balance_threshold
        
        # 计算不平衡比例
        max_count = target_counts.max()
        min_count = target_counts.min()
        imbalance_ratio = max_count / min_count
        
        print(f"\n⚖️ 平衡性分析:")
        print(f"   少数类占比: {min_class_ratio:.3f}")
        print(f"   不平衡比例: {imbalance_ratio:.2f}:1")
        print(f"   平衡阈值: {balance_threshold}")
        
        if is_balanced:
            print(f"   ✅ 数据集相对平衡 (少数类占比 >= {balance_threshold})")
            recommended_strategy = "balanced"
        else:
            print(f"   ⚠️ 数据集不平衡 (少数类占比 < {balance_threshold})")
            recommended_strategy = "imbalanced"
        
        print(f"\n💡 推荐策略: {recommended_strategy}")
        
        return {
            'total_samples': total_samples,
            'feature_count': len(df.columns) - 1,
            'target_distribution': target_counts.to_dict(),
            'min_class_ratio': min_class_ratio,
            'imbalance_ratio': imbalance_ratio,
            'is_balanced': is_balanced,
            'recommended_strategy': recommended_strategy,
            'balance_threshold': balance_threshold
        }
        
    except Exception as e:
        print(f"❌ 分析数据集时出错: {e}")
        raise


def run_balanced_feature_selection(data_path: str,
                                 target_col: str,
                                 methods: List[str] = None,
                                 top_k: int = 50,
                                 output_dir: str = "balanced_feature_selection",
                                 id_col: str = None) -> Dict[str, str]:
    """
    对平衡数据集进行特征选择（不需要重采样）
    
    Args:
        data_path: 数据文件路径
        target_col: 目标列名
        methods: 特征选择方法列表
        top_k: 选择的特征数量
        output_dir: 输出目录
        id_col: ID列名
        
    Returns:
        Dict: 各方法的结果文件路径
    """
    print("\n🎯 开始平衡数据集特征选择...")
    print("=" * 60)
    
    if methods is None:
        methods = ['randomforest', 'lightgbm', 'elasticnet', 'fcbf', 'relief']
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print(f"📁 加载数据: {data_path}")
    X, y, feature_labels = acquire_data_for_feature_selection(data_path, target_col, id_col)
    
    print(f"📊 数据维度: {X.shape}")
    print(f"🎯 目标分布: {y.value_counts().to_dict()}")
    
    # 可用的方法函数映射
    method_functions = {
        'lightgbm': lightgbm_feature_selection,
        'randomforest': randomforest_feature_selection,
        'elasticnet': elasticnet_feature_selection,
        'fcbf': fcbf_feature_selection,
        'relief': relief_feature_selection
    }
    
    # 过滤可用方法
    available_methods = []
    for method in methods:
        if method == 'lightgbm' and not LIGHTGBM_AVAILABLE:
            print(f"⚠️ 跳过 {method}: LightGBM 不可用")
            continue
        elif method == 'fcbf' and not FCBF_AVAILABLE:
            print(f"⚠️ 跳过 {method}: FCBF 不可用")
            continue
        elif method == 'relief' and not RELIEF_AVAILABLE:
            print(f"⚠️ 跳过 {method}: ReliefF 不可用")
            continue
        elif method == 'mrmr' and not PYMRMR_AVAILABLE:
            print(f"⚠️ 跳过 {method}: mRMR 不可用")
            continue
        elif method in method_functions or method == 'mrmr':
            available_methods.append(method)
        else:
            print(f"⚠️ 跳过 {method}: 未知方法")
    
    if not available_methods:
        raise ValueError("没有可用的特征选择方法")
    
    print(f"✅ 可用方法: {available_methods}")
    
    # 执行特征选择
    results = {}
    
    for method in available_methods:
        print(f"\n🔄 执行 {method.upper()} 特征选择...")
        
        try:
            if method == 'mrmr':
                # mRMR需要完整数据框
                df = pd.read_csv(data_path)
                if id_col and id_col in df.columns:
                    df = df.drop(id_col, axis=1)
                elif not id_col:
                    df = df.drop(df.columns[0], axis=1)
                
                result = mrmr_feature_selection(df, target_col, 1, top_k)
            else:
                # 其他方法
                result = method_functions[method](X, y, feature_labels, 1, top_k)
            
            # 保存结果
            output_file = output_path / f"{method}_features.csv"
            save_feature_selection_results(result, str(output_file))
            results[method] = str(output_file)
            
            # 显示选择的特征数量
            if isinstance(result, pd.DataFrame):
                feature_count = len(result)
            elif isinstance(result, pd.Series):
                feature_count = len(result)
            else:
                feature_count = "未知"
            
            print(f"   ✅ {method}: 选择了 {feature_count} 个特征")
            print(f"   💾 保存到: {output_file}")
            
        except Exception as e:
            print(f"   {method} 执行失败: {e}")
            continue
    
    print(f"\n 平衡数据集特征选择完成!")
    print(f" 结果保存在: {output_dir}")
    
    return results


def run_auto_feature_selection(data_path: str,
                               target_col: str,
                               methods: List[str] = None,
                               top_k: int = 50,
                               output_dir: Optional[str] = None,
                               id_col: Optional[str] = None,
                               balance_threshold: float = 0.3,
                               n_iterations: int = 100,
                               sample_ratio: float = 0.75,
                               match_ratio: int = 3,
                               random_state: int = 123,
                               match_cols: Optional[List[str]] = None,
                               run_robust_analysis: bool = True,
                               robust_threshold: float = 0.5,
                               run_feature_voting: bool = True,
                               run_final_dataset: bool = True,
                               final_min_features: int = 10,
                               final_covariates: Optional[List[str]] = None) -> Dict[str, str]:
    """根据数据分布自动选择平衡或重采样特征选择策略。

    当数据集相对平衡时，调用 ``run_balanced_feature_selection``；
    当数据集不平衡时，自动推断匹配变量并调用重采样流水线：
        ``generate_resampled_datasets`` + ``run_feature_selection_methods``。
    """

    # 分析数据集平衡性
    balance_info = analyze_dataset_balance(
        data_path=data_path,
        target_col=target_col,
        balance_threshold=balance_threshold,
    )

    strategy = balance_info.get("recommended_strategy", "balanced")
    print(f"\n Auto feature selection strategy: {strategy}")

    # 平衡数据：直接走原有的平衡特征选择流程
    if strategy == "balanced":
        return run_balanced_feature_selection(
            data_path=data_path,
            target_col=target_col,
            methods=methods,
            top_k=top_k,
            output_dir=output_dir or get_feature_selection_subdir("balanced"),
            id_col=id_col,
        )

    # 不平衡数据：使用重采样特征选择
    print("\n 检测到数据集不平衡，启用重采样特征选择流水线...")

    # 匹配变量设置：
    # - 如果用户显式提供 match_cols，则使用用户提供的列（并检查是否存在）；
    # - 如果未提供，则使用默认 ['age', 'bmi', 'gender']，只保留在数据中真实存在的列。
    try:
        df = pd.read_csv(data_path)
        available_cols = set(df.columns)

        if match_cols is None:
            default_match_cols = ["age", "bmi", "gender"]
            match_cols = [c for c in default_match_cols if c in available_cols]
            if not match_cols:
                raise ValueError(
                    "未能在数据中找到任何默认匹配变量 ['age', 'bmi', 'gender']，"
                    "请在 run_auto_feature_selection 中显式设置 match_cols 参数。"
                )
            print(
                f"Using default match columns (count={len(match_cols)}): "
                f"{match_cols}"
            )
        else:
            # 用户自定义匹配变量：只保留真实存在的列
            original_match_cols = list(match_cols)
            match_cols = [c for c in original_match_cols if c in available_cols]
            if not match_cols:
                raise ValueError(
                    f"提供的匹配变量 {original_match_cols} 在数据中均不存在，"
                    "请检查列名或重新设置 match_cols。"
                )
            missing = [c for c in original_match_cols if c not in available_cols]
            if missing:
                print(f" 以下匹配变量在数据中未找到，将被忽略: {missing}")
            print(
                f"Using user-specified match columns (count={len(match_cols)}): "
                f"{match_cols}"
            )
    except Exception as e:
        print(f" 匹配变量配置失败: {e}")
        raise

    # 生成重采样数据集
    _ = generate_resampled_datasets(
        data_path=data_path,
        n_iterations=n_iterations,
        target_col=target_col,
        match_cols=match_cols,
        sample_ratio=sample_ratio,
        match_ratio=match_ratio,
        random_state=random_state,
    )

    resampling_dir = get_feature_selection_subdir("resampling")

    # 在重采样数据上运行特征选择方法
    fs_summary = run_feature_selection_methods(
        input_dir=resampling_dir,
        target_col=target_col,
        iterations=n_iterations,
        methods=methods,
        top_k=top_k,
        output_base_dir=output_dir or get_feature_selection_dir(),
        id_col=id_col,
    )

    # run_feature_selection_methods 返回的是摘要，这里将其转换为与
    # run_balanced_feature_selection 类似的返回形式：方法名 -> 输出目录
    results: Dict[str, str] = {}
    for method, info in fs_summary.items():
        if isinstance(info, dict) and "output_dir" in info:
            results[method] = info["output_dir"]
        else:
            results[method] = str(info)

    base_results_dir = output_dir or get_feature_selection_dir()

    # 可选：稳健特征分析
    robust_results = None
    if run_robust_analysis:
        robust_output_dir = os.path.join(base_results_dir, "robust_features_analysis")
        robust_results = analyze_robust_features(
            results_base_dir=base_results_dir,
            threshold=robust_threshold,
            output_dir=robust_output_dir,
        )

    # 可选：特征投票分析（依赖稳健特征分析）
    voting_results = None
    voting_output_dir = None
    if run_feature_voting:
        voting_output_dir = os.path.join(base_results_dir, "feature_voting_analysis")
        voting_results = analyze_feature_voting(
            results_base_dir=base_results_dir,
            threshold=robust_threshold,
            output_dir=voting_output_dir,
        )

    # 可选：根据投票结果创建最终特征数据集
    if run_final_dataset:
        analysis_dir = voting_output_dir or os.path.join(base_results_dir, "feature_voting_analysis")
        final_output_path = os.path.join(base_results_dir, "final_feature_dataset.csv")
        _ = create_final_feature_dataset(
            original_data_path=data_path,
            analysis_dir=analysis_dir,
            output_path=final_output_path,
            min_features=final_min_features,
            covariates=final_covariates,
            target_col=target_col,
            id_col=id_col,
        )

    print("\n 自动特征选择（含重采样、稳健特征与投票分析）完成!")
    print(f" 结果基础目录: {base_results_dir}")

    return results


def sample_extract(data: pd.DataFrame, 
                  target_col: str,
                  match_cols: list,
                  sample_ratio: float,
                  match_ratio: int,
                  random_state: int) -> pd.DataFrame:
    """
    Extract matched samples from data using propensity score matching
    
    Args:
        data: Input DataFrame
        target_col: Target column name for matching
        match_cols: Columns to use for matching
        sample_ratio: Ratio of data to sample
        match_ratio: Matching ratio
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with matched samples
    """
        
    np.random.seed(random_state)
    
    # 1. Random sampling (equivalent to R's sample())
    n_samples = int(len(data) * sample_ratio)
    sample_indices = np.random.choice(data.index, size=n_samples, replace=False)
    data_sample = data.loc[sample_indices].copy()
    
    # 2. Propensity score matching using nearest neighbors
    # Separate treatment and control groups
    treatment_group = data_sample[data_sample[target_col] == 1]
    control_group = data_sample[data_sample[target_col] == 0]
    
    if len(treatment_group) == 0 or len(control_group) == 0:
        print("Warning: One of the groups is empty after sampling")
        return data_sample
    
    # Standardize matching variables
    scaler = StandardScaler()
    
    # Fit scaler on all data and transform both groups
    all_match_data = data_sample[match_cols].fillna(data_sample[match_cols].mean())
    scaler.fit(all_match_data)
    
    treatment_features = scaler.transform(
        treatment_group[match_cols].fillna(treatment_group[match_cols].mean())
    )
    control_features = scaler.transform(
        control_group[match_cols].fillna(control_group[match_cols].mean())
    )
    
    # Use KNN for matching (equivalent to R's matchit with nearest neighbor)
    nn = NearestNeighbors(n_neighbors=min(match_ratio, len(control_group)), 
                         metric='euclidean')
    nn.fit(control_features)
    
    # Find matches for each treatment case
    matched_indices = []
    matched_indices.extend(treatment_group.index.tolist())  # Include all treatment cases
    
    distances, indices = nn.kneighbors(treatment_features)
    
    # Add matched control cases
    for i, treatment_idx in enumerate(treatment_group.index):
        for j in range(min(match_ratio, len(indices[i]))):
            control_idx = control_group.index[indices[i][j]]
            if control_idx not in matched_indices:  # Avoid duplicates
                matched_indices.append(control_idx)
    
    # Create matched dataset
    matched_data = data_sample.loc[matched_indices].copy()
    
    # 3. Remove last 3 columns (equivalent to R's [-c(92,93,94)])
    # Note: This removes the last 3 columns regardless of their position
    if len(matched_data.columns) >= 3:
        matched_data = matched_data.iloc[:, :-3]
    
    return matched_data


def check_data_columns(data_path: str, target_col: str, match_cols: list) -> bool:
    """
    Check if the specified columns exist in the data
    
    Args:
        data_path: Path to data file
        target_col: Target column name
        match_cols: List of matching column names
        
    Returns:
        True if all columns exist, False otherwise
    """
    try:
        data = pd.read_csv(data_path)
        print(f"Data shape: {data.shape}")
        print(f"Available columns: {list(data.columns)[:10]}...")  # Show first 10 columns
        
        missing_cols = []
        
        # Check target column
        if target_col not in data.columns:
            missing_cols.append(target_col)
        else:
            print(f"✓ Target column '{target_col}' found")
            print(f"  Value counts: {data[target_col].value_counts().to_dict()}")
        
        # Check match columns
        for col in match_cols:
            if col not in data.columns:
                missing_cols.append(col)
            else:
                print(f"✓ Match column '{col}' found")
        
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            print("Available columns:")
            for i, col in enumerate(data.columns):
                print(f"  {i+1:2d}. {col}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking data: {e}")
        return False


def generate_resampled_datasets(data_path: str,
                              n_iterations: int,
                              target_col: str,
                              match_cols: list,
                              sample_ratio: float,
                              match_ratio: int,
                              random_state: int) -> list:
    """
    Generate multiple resampled and matched datasets for PLSDA analysis
    
    Args:
        data_path: Path to input CSV file
        n_iterations: Number of iterations
        target_col: Target column for matching
        match_cols: Columns to use for matching
        sample_ratio: Sampling ratio
        match_ratio: Matching ratio
        random_state: Base random seed
        
    Returns:
        List of file paths for generated datasets
        
    Note:
        Output directory is managed by ClinMetML path manager
    """
        
    # Read data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = pd.read_csv(data_path)
    print(f"Loaded data with shape: {data.shape}")
    
    # Create output directory using path manager
    output_path = Path(get_feature_selection_subdir("resampling"))
    
    # Store results
    result_files = []
    
    print(f"Generating {n_iterations} resampled datasets...")
    
    for i in range(1, n_iterations + 1):
        try:
            # Use different random state for each iteration
            current_seed = random_state + i
            
            # Extract matched sample
            data_sample = sample_extract(
                data=data,
                target_col=target_col,
                match_cols=match_cols,
                sample_ratio=sample_ratio,
                match_ratio=match_ratio,
                random_state=current_seed
            )
            
            # Save file
            file_name = f"matched_Data_test{i}.csv"
            file_path = output_path / file_name
            data_sample.to_csv(file_path, index=False)
            result_files.append(str(file_path))
            
            # Progress indicator
            if i % 100 == 0:
                print(f"Completed {i}/{n_iterations} iterations")
                
        except Exception as e:
            print(f"Error in iteration {i}: {e}")
            continue
    
    print(f"✓ Generated {len(result_files)} datasets in 'resampling' directory")
    return result_files


def run_feature_selection_pipeline(data_path: str,
                                 n_iterations: int,
                                 target_col: str,
                                 match_cols: list,
                                 sample_ratio: float = 0.75,
                                 match_ratio: int = 3,
                                 random_state: int = 123) -> bool:
    """
    Main function to run the complete feature selection pipeline
    
    Args:
        data_path: Path to input data file
        n_iterations: Number of resampling iterations
        target_col: Target column for matching
        match_cols: Columns to use for matching
        sample_ratio: Sampling ratio (default: 0.75)
        match_ratio: Matching ratio (default: 3)
        random_state: Random seed (default: 123)
        
    Returns:
        True if successful, False otherwise
        
    Note:
        Output directory is managed by ClinMetML path manager
    """
        
    try:
        print("="*60)
        print("Feature Selection Pipeline")
        print("="*60)
        
        # Check if data file exists
        if not os.path.exists(data_path):
            print(f"❌ Data file not found: {data_path}")
            return False
        
        # Check data columns
        print(f"\nChecking data columns...")
        if not check_data_columns(data_path, target_col, match_cols):
            return False
        
        # Generate resampled datasets
        result_files = generate_resampled_datasets(
            data_path=data_path,
            n_iterations=n_iterations,
            target_col=target_col,
            match_cols=match_cols,
            sample_ratio=sample_ratio,
            match_ratio=match_ratio,
            random_state=random_state
        )
        
        if len(result_files) > 0:
            print(f"\n✅ Pipeline completed successfully!")
            print(f"Generated {len(result_files)} matched datasets")
            print(f"Output directory: {get_feature_selection_subdir('resampling')}")
            return True
        else:
            print(f"\n❌ Pipeline failed - no datasets generated")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        return False


class FeatureSelector:
    """High-level feature selection interface used by ClinMetMLPipeline.

    This class provides a `select_features` method compatible with the
    pipeline configuration shown in the README and examples, but internally
    it reuses the `run_auto_feature_selection` pipeline implemented above.
    """

    def __init__(self):
        self.last_results_dir: Optional[str] = None

    def select_features(
        self,
        data: pd.DataFrame,
        target_column: str,
        **kwargs,
    ) -> pd.DataFrame:
        """Select features and return a DataFrame containing selected features.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe including the target column.
        target_column : str
            Name of the target column in `data`.
        **kwargs : dict
            Additional configuration, typically coming from
            `pipeline_config['feature_selection']`, e.g.:

            - method: list[str] or str
            - k_best: mapped to `top_k`
            - output_dir: base directory for feature selection outputs
            - id_column: identifier column name (mapped to `id_col`)
            - other arguments accepted by `run_auto_feature_selection`.
        """

        # Resolve output directory
        output_dir = kwargs.get("output_dir")
        if output_dir is None:
            output_dir = get_feature_selection_dir()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Persist current data to CSV for the file-based APIs
        input_csv = output_path / "feature_selection_input.csv"
        data.to_csv(input_csv, index=False)

        # Map generic pipeline params to run_auto_feature_selection arguments
        methods = kwargs.get("method")
        # Allow method to be a single string or a list
        if isinstance(methods, str):
            methods_arg = [methods]
        else:
            methods_arg = methods

        top_k = kwargs.get("k_best", kwargs.get("top_k", 50))
        id_col = kwargs.get("id_column", kwargs.get("id_col"))

        # Optional advanced parameters with sensible defaults
        balance_threshold = kwargs.get("balance_threshold", 0.3)
        n_iterations = kwargs.get("n_iterations", 3)
        sample_ratio = kwargs.get("sample_ratio", 0.75)
        match_ratio = kwargs.get("match_ratio", 3)
        random_state = kwargs.get("random_state", 123)
        match_cols = kwargs.get("match_cols")
        final_min_features = kwargs.get("final_min_features", 10)
        final_covariates = kwargs.get("final_covariates")

        results = run_auto_feature_selection(
            data_path=str(input_csv),
            target_col=target_column,
            methods=methods_arg,
            top_k=top_k,
            output_dir=str(output_path),
            id_col=id_col,
            balance_threshold=balance_threshold,
            n_iterations=n_iterations,
            sample_ratio=sample_ratio,
            match_ratio=match_ratio,
            random_state=random_state,
            match_cols=match_cols,
            run_robust_analysis=True,
            robust_threshold=kwargs.get("robust_threshold", 0.5),
            run_feature_voting=True,
            run_final_dataset=True,
            final_min_features=final_min_features,
            final_covariates=final_covariates,
        )

        # By convention, run_auto_feature_selection will write a
        # "final_feature_dataset.csv" into the base results directory.
        self.last_results_dir = str(output_path)
        final_dataset_path = output_path / "final_feature_dataset.csv"

        if not final_dataset_path.exists():
            raise FileNotFoundError(
                f"Expected final feature dataset at '{final_dataset_path}', "
                "but the file was not found. Please check feature selection outputs."
            )

        selected_data = pd.read_csv(final_dataset_path)
        return selected_data


# ==================== 特征选择方法 ====================

def acquire_data_for_feature_selection(file_path: str, target_col: str, id_col: str = None) -> tuple:
    """
    加载数据并准备特征和标签（用于特征选择）
    
    Args:
        file_path: CSV文件路径
        target_col: 目标列名（用户定义的target-col）
        id_col: ID列名（如果为None，自动删除第一列）
        
    Returns:
        tuple: (X, y, feature_labels)
    """
    try:
        # 使用更高效的读取方式，指定数据类型以节省内存
        df = pd.read_csv(file_path, low_memory=False, dtype='float32')
        
        # 删除ID列
        if id_col and id_col in df.columns:
            df = df.drop(id_col, axis=1)
        elif not id_col:
            # 删除第一列（假设是ID列）
            df = df.drop(df.columns[0], axis=1)
        
        # 提取目标变量和特征
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # 直接提取，避免额外的copy操作
        y = df[target_col].astype('int32')
        X = df.drop(target_col, axis=1)
        feature_labels = X.columns.values
        
        # 清理内存
        del df
        
        return X, y, feature_labels
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise


def lightgbm_feature_selection(X: pd.DataFrame, y: pd.Series, 
                              feature_labels: np.ndarray, 
                              iteration: int,
                              top_k: int = 50,
                              test_size: float = 0.3,
                              random_state: int = 42) -> pd.DataFrame:
    """
    使用LightGBM进行特征选择
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not available")
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'verbose': -1
    }
    
    lgb_train = lgb.Dataset(X_train, y_train)
    model = lgb.train(params, lgb_train, num_boost_round=100)
    
    feature_importance = model.feature_importance(importance_type='gain')
    indices = feature_importance.argsort()[::-1]
    top_indices = indices[:top_k]
    
    importances_df = pd.DataFrame({
        'feature': feature_labels[top_indices],
        'importance': feature_importance[top_indices]
    })
    
    print(f"LightGBM - 已完成: {iteration}")
    return importances_df


def randomforest_feature_selection(X: pd.DataFrame, y: pd.Series,
                                  feature_labels: np.ndarray,
                                  iteration: int,
                                  top_k: int = 50,
                                  test_size: float = 0.3,
                                  random_state: int = 42,
                                  n_estimators: int = 1000) -> pd.DataFrame:
    """
    使用随机森林进行特征选择
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    forest = RandomForestClassifier(
        n_estimators=n_estimators, 
        random_state=random_state, 
        n_jobs=-1
    )
    forest.fit(X_train, y_train)
    
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:top_k]
    
    importances_df = pd.DataFrame({
        'feature': feature_labels[top_indices],
        'importance': importances[top_indices]
    })
    
    print(f"RandomForest - 已完成: {iteration}")
    return importances_df


def elasticnet_feature_selection(X: pd.DataFrame, y: pd.Series,
                                feature_labels: np.ndarray,
                                iteration: int,
                                top_k: int = 50,
                                test_size: float = 0.3,
                                random_state: int = 42,
                                alpha: float = 0.005,
                                l1_ratio: float = 0.1) -> pd.DataFrame:
    """
    使用ElasticNet进行特征选择
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    enet.fit(X_train, y_train)
    
    coef = enet.coef_
    importances = np.abs(coef)
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:top_k]
    
    importances_df = pd.DataFrame({
        'feature': feature_labels[top_indices],
        'importance': importances[top_indices]
    })
    
    print(f"ElasticNet - 已完成: {iteration}")
    return importances_df


def fcbf_feature_selection(X: pd.DataFrame, y: pd.Series,
                          feature_labels: np.ndarray,
                          iteration: int,
                          top_k: int = 50,
                          test_size: float = 0.3,
                          random_state: int = 42) -> pd.DataFrame:
    """
    使用FCBF进行特征选择
    """
    if not FCBF_AVAILABLE:
        raise ImportError("FCBF module not available")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    X_train_array = np.array(X_train)
    y_train_array = np.array(y_train)
    
    # 根据可用的FCBF包类型选择使用方式
    if FCBF_TYPE == 'skfeature':
        # 使用 skfeature 的 FCBF
        # 临时修复NumPy兼容性问题
        original_zeros = np.zeros
        def patched_zeros(shape, dtype=float, order='C', **kwargs):
            # 处理可能的dtypes参数（某些版本可能使用dtypes而不是dtype）
            if 'dtypes' in kwargs:
                dtype = kwargs.pop('dtypes')
            return original_zeros(shape, dtype=dtype, order=order, **kwargs)
        np.zeros = patched_zeros
        
        try:
            from skfeature.function.information_theoretical_based import FCBF
            result = FCBF.fcbf(X_train_array, y_train_array, n_selected_features=top_k)
        finally:
            # 恢复原始函数
            np.zeros = original_zeros
        
        # FCBF返回(selected_features, selected_feature_scores)
        if isinstance(result, tuple) and len(result) == 2:
            idx, scores = result
            # 确保索引是整数类型
            idx = [int(i) for i in idx[:top_k]]
        else:
            # 如果返回格式不同，使用前top_k个特征作为fallback
            idx = list(range(min(top_k, len(feature_labels))))
        
        selected_features = feature_labels[idx]
        importances_df = pd.DataFrame({'feature': selected_features})
        
    elif FCBF_TYPE == 'fcbf':
        # 使用 fcbf 包
        from fcbf import fcbf
        # fcbf包需要DataFrame和Series，返回相关特征列表
        X_train_df = pd.DataFrame(X_train, columns=feature_labels)
        relevant_features, irrelevant_features, correlations = fcbf(X_train_df, y_train)
        
        # 取前top_k个特征
        selected_features = relevant_features[:top_k]
        importances_df = pd.DataFrame({'feature': selected_features})
        
    elif FCBF_TYPE == 'FCBF_module':
        # 使用 FCBF_module 包
        # 临时修复NumPy兼容性问题
        original_zeros = np.zeros
        def patched_zeros(shape, dtype=float, order='C', **kwargs):
            # 处理可能的dtypes参数（某些版本可能使用dtypes而不是dtype）
            if 'dtypes' in kwargs:
                dtype = kwargs.pop('dtypes')
            return original_zeros(shape, dtype=dtype, order=order, **kwargs)
        np.zeros = patched_zeros
        
        try:
            from FCBF_module import FCBF
            fcbf_selector = FCBF()
            idx = fcbf_selector.fcbf(X_train_array, y_train_array, n_selected_features=top_k)
            idx = idx[:top_k]
        finally:
            # 恢复原始函数
            np.zeros = original_zeros
        
        selected_features = feature_labels[idx]
        importances_df = pd.DataFrame({'feature': selected_features})
        
    else:
        # 如果没有可用的FCBF包，使用fallback
        print(f"Warning: No FCBF package available, using first {top_k} features")
        idx = list(range(min(top_k, len(feature_labels))))
        selected_features = feature_labels[idx]
        importances_df = pd.DataFrame({'feature': selected_features})
    
    print(f"FCBF - 已完成: {iteration}")
    return importances_df


def relief_feature_selection(X: pd.DataFrame, y: pd.Series,
                           feature_labels: np.ndarray,
                           iteration: int,
                           top_k: int = 50,
                           test_size: float = 0.3,
                           random_state: int = 42,
                           n_neighbors: int = 10) -> pd.DataFrame:
    """
    使用ReliefF进行特征选择
    """
    if not RELIEF_AVAILABLE:
        raise ImportError("skrebate not available")
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    X_train_array = np.array(X_train)
    y_train_array = np.array(y_train)
    
    # 导入ReliefF（只有在可用时才会执行到这里）
    from skrebate import ReliefF
    fs = ReliefF(n_neighbors=n_neighbors)
    fs.fit(X_train_array, y_train_array)
    
    importances = fs.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:top_k]
    
    importances_df = pd.DataFrame({
        'feature': feature_labels[top_indices],
        'importance': importances[top_indices]
    })
    
    print(f"ReliefF - 已完成: {iteration}")
    return importances_df


def mrmr_feature_selection(df: pd.DataFrame,
                          target_col: str,
                          iteration: int,
                          top_k: int = 50) -> pd.Series:
    """
    使用mRMR进行特征选择
    """
    if not PYMRMR_AVAILABLE:
        raise ImportError("pymrmr module not available")
    
    # mRMR需要完整的数据框
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    if not df.columns.is_unique:
        raise ValueError("Duplicate column names are not supported")

    predictors = [c for c in df.columns if c != target_col]
    if not predictors:
        raise ValueError("No predictor columns available")

    ordered_df = df.loc[:, [target_col] + predictors].copy()

    mr = pymrmr.mRMR(
        ordered_df, 'MIQ', min(top_k, len(predictors))
    )

    if not set(mr).issubset(predictors):
        raise RuntimeError("mRMR returned invalid predictor names")
    importances_df = pd.Series(mr)
    
    print(f"mRMR - 已完成: {iteration}")
    return importances_df


def save_feature_selection_results(results_df, filename: str):
    """
    保存特征选择结果到文件
    """
    # 确保输出目录存在
    output_dir = Path(filename).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理不同类型的结果
    if isinstance(results_df, pd.DataFrame):
        if results_df.empty:
            print(f"Warning: Empty DataFrame for {filename}")
            # 创建一个空的CSV文件
            pd.DataFrame(columns=['feature']).to_csv(filename, index=False)
        else:
            results_df.to_csv(filename, index=False)
    elif isinstance(results_df, pd.Series):
        # 对于mRMR等返回Series的方法
        results_df.to_csv(filename, index=False, header=['feature'])
    else:
        print(f"Warning: Unknown result type {type(results_df)} for {filename}")
        # 尝试转换为DataFrame
        try:
            pd.DataFrame(results_df).to_csv(filename, index=False)
        except Exception as e:
            print(f"Error saving {filename}: {e}")
            # 创建空文件
            pd.DataFrame(columns=['feature']).to_csv(filename, index=False)


def run_feature_selection_methods(input_dir: str = "resampling",
                                target_col: str = "dep_5",
                                iterations: int = 1000,
                                methods: list = None,
                                top_k: int = 50,
                                output_base_dir: str = "feature_selection_results",
                                id_col: str = None) -> dict:
    """
    运行所有特征选择方法
    
    Args:
        input_dir: 输入数据目录（resampling文件夹）
        target_col: 目标列名（用户定义的target-col）
        iterations: 迭代次数（与feature_selection.py中的iterations一致）
        methods: 要运行的方法列表
        top_k: 每个方法选择的特征数量（用户可定义）
        output_base_dir: 输出基础目录
        id_col: ID列名（第一列，通常是eid）
        
    Returns:
        dict: 每个方法的结果统计
    """
    if methods is None:
        methods = ['randomforest', 'elasticnet']
        if LIGHTGBM_AVAILABLE:
            methods.append('lightgbm')
        if RELIEF_AVAILABLE:
            methods.append('relief')
        if FCBF_AVAILABLE:
            methods.append('fcbf')
        if PYMRMR_AVAILABLE:
            methods.append('mrmr')
    
    # 检查方法可用性
    available_methods = ['randomforest', 'elasticnet']
    if LIGHTGBM_AVAILABLE:
        available_methods.append('lightgbm')
    if RELIEF_AVAILABLE:
        available_methods.append('relief')
    if FCBF_AVAILABLE:
        available_methods.append('fcbf')
    if PYMRMR_AVAILABLE:
        available_methods.append('mrmr')
    
    # 过滤不可用的方法
    original_methods = methods.copy()
    methods = [m for m in methods if m in available_methods]
    
    # 显示被过滤掉的方法
    filtered_out = [m for m in original_methods if m not in methods]
    if filtered_out:
        print(f"⚠️  以下方法不可用，已跳过: {', '.join(filtered_out)}")
        print(f"💡 可用的方法: {', '.join(available_methods)}")
    
    if not methods:
        print("❌ 指定的所有方法都不可用，使用默认方法: randomforest, elasticnet")
        methods = ['randomforest', 'elasticnet']
    else:
        print(f"✅ 将运行以下方法: {', '.join(methods)}")
    
    # 方法映射
    method_functions = {
        'lightgbm': lightgbm_feature_selection,
        'randomforest': randomforest_feature_selection,
        'elasticnet': elasticnet_feature_selection,
        'fcbf': fcbf_feature_selection,
        'relief': relief_feature_selection,
        'mrmr': mrmr_feature_selection
    }
    
    results_summary = {}
    
    for method in methods:
        if method not in method_functions:
            print(f"Warning: Unknown method '{method}', skipping...")
            continue
            
        print(f"\n开始运行 {method.upper()} 特征选择...")
        method_results = []
        
        # 创建方法特定的输出目录
        method_output_dir = Path(output_base_dir) / f"res_{method}_{iterations}"
        method_output_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(1, iterations + 1):
            input_file = Path(input_dir) / f"matched_Data_test{i}.csv"
            
            if not input_file.exists():
                print(f"Warning: File {input_file} not found, skipping...")
                continue
            
            # 显示进度
            if i % 10 == 0 or i <= 5:
                print(f"  处理文件 {i}/{iterations}...")
            
            try:
                if method == 'mrmr':
                    # mRMR需要完整的数据框
                    df = pd.read_csv(input_file)
                    if id_col:
                        df = df.drop(id_col, axis=1)
                    else:
                        df = df.drop(df.columns[0], axis=1)
                    
                    results = mrmr_feature_selection(df, target_col, i, top_k)
                else:
                    # 其他方法
                    X, y, feature_labels = acquire_data_for_feature_selection(input_file, target_col, id_col)
                    results = method_functions[method](X, y, feature_labels, i, top_k)
                
                # 保存结果
                output_file = method_output_dir / f"{method}_test_{i}.csv"
                if results is not None:
                    save_feature_selection_results(results, str(output_file))
                    method_results.append(str(output_file))
                else:
                    print(f"Warning: No results returned for {input_file} with {method}")
                
            except Exception as e:
                print(f"Error processing {input_file} with {method}: {e}")
                continue
        
        results_summary[method] = {
            'completed_files': len(method_results),
            'output_dir': str(method_output_dir)
        }
        
        print(f"{method.upper()} 完成: {len(method_results)}/{iterations} 文件")
    
    return results_summary


def analyze_robust_features(results_base_dir: str = "feature_selection_results", 
                          threshold: float = 0.5,
                          output_dir: str = "robust_features_analysis") -> Dict[str, Dict]:
    """
    分析每个特征选择方法的稳健特征
    
    Args:
        results_base_dir: 特征选择结果的基础目录
        threshold: 特征被选中的频率阈值 (0-1之间)
        output_dir: 输出目录
        
    Returns:
        Dict: 每个方法的稳健特征统计结果
    """
    print("🔍 开始分析稳健特征...")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有方法的结果目录
    method_dirs = glob.glob(os.path.join(results_base_dir, "res_*"))
    
    if not method_dirs:
        print("❌ 未找到特征选择结果目录")
        return {}
    
    robust_features_results = {}
    
    for method_dir in method_dirs:
        # 提取方法名称
        method_name = os.path.basename(method_dir).replace("res_", "").split("_")[0]
        
        print(f"\n📊 分析方法: {method_name}")
        print(f"   目录: {method_dir}")
        
        # 获取该方法的所有CSV文件
        csv_files = glob.glob(os.path.join(method_dir, "*.csv"))
        
        if not csv_files:
            print(f"   ⚠️  未找到CSV文件")
            continue
            
        print(f"   📁 找到 {len(csv_files)} 个结果文件")
        
        # 统计特征频率
        feature_counter = Counter()
        total_iterations = 0
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if 'feature' in df.columns:
                    features = df['feature'].dropna().tolist()
                    feature_counter.update(features)
                    total_iterations += 1
                else:
                    print(f"   ⚠️  文件 {os.path.basename(csv_file)} 缺少 'feature' 列")
            except Exception as e:
                print(f"   ❌ 读取文件 {os.path.basename(csv_file)} 失败: {e}")
        
        if total_iterations == 0:
            print(f"   ❌ 没有有效的迭代结果")
            continue
            
        # 计算特征频率并筛选稳健特征
        feature_frequencies = {}
        robust_features = []
        
        for feature, count in feature_counter.items():
            frequency = count / total_iterations
            feature_frequencies[feature] = {
                'count': count,
                'frequency': frequency,
                'total_iterations': total_iterations
            }
            
            if frequency >= threshold:
                robust_features.append(feature)
        
        # 按频率排序
        sorted_features = sorted(feature_frequencies.items(), 
                               key=lambda x: x[1]['frequency'], 
                               reverse=True)
        
        print(f"   📈 总特征数: {len(feature_frequencies)}")
        print(f"   🎯 稳健特征数 (频率 >= {threshold}): {len(robust_features)}")
        
        # 保存结果
        robust_features_results[method_name] = {
            'total_iterations': total_iterations,
            'total_features': len(feature_frequencies),
            'robust_features_count': len(robust_features),
            'robust_features': robust_features,
            'all_feature_frequencies': feature_frequencies,
            'threshold': threshold
        }
        
        # 保存详细统计到CSV
        stats_df = pd.DataFrame([
            {
                'feature': feature,
                'count': stats['count'],
                'frequency': stats['frequency'],
                'is_robust': stats['frequency'] >= threshold
            }
            for feature, stats in sorted_features
        ])
        
        stats_file = os.path.join(output_dir, f"{method_name}_feature_statistics.csv")
        stats_df.to_csv(stats_file, index=False)
        print(f"   💾 统计结果保存到: {stats_file}")
        
        # 保存稳健特征列表
        if robust_features:
            robust_df = pd.DataFrame({'robust_feature': robust_features})
            robust_file = os.path.join(output_dir, f"{method_name}_robust_features.csv")
            robust_df.to_csv(robust_file, index=False)
            print(f"   💾 稳健特征保存到: {robust_file}")
    
    # 保存汇总报告
    summary_data = []
    for method, results in robust_features_results.items():
        summary_data.append({
            'method': method,
            'total_iterations': results['total_iterations'],
            'total_features': results['total_features'],
            'robust_features_count': results['robust_features_count'],
            'robust_ratio': results['robust_features_count'] / results['total_features'] if results['total_features'] > 0 else 0,
            'threshold': results['threshold']
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_dir, "robust_features_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"\n📋 汇总报告保存到: {summary_file}")
    
    print(f"\n✅ 稳健特征分析完成! 结果保存在: {output_dir}")
    return robust_features_results



def create_final_feature_dataset(original_data_path: str,
                               analysis_dir: str,
                               output_path: str = "feature_selection_outputs/final_dataset.csv",
                               min_features: int = 10,
                               covariates: List[str] = None,
                               target_col: str = None,
                               id_col: str = None) -> bool:
    """
    根据投票分析结果创建最终的特征筛选数据集
    
    Args:
        original_data_path: 原始数据文件路径
        analysis_dir: 分析结果目录路径
        output_path: 输出文件路径
        min_features: 最少特征数量阈值
        covariates: 需要保留的协变量列表
        target_col: 目标列名
        id_col: ID列名
        
    Returns:
        True if successful, False otherwise
    """
    
    try:
        print(f"\n🎯 开始创建最终特征数据集...")
        print(f"📁 原始数据: {original_data_path}")
        print(f"📊 分析目录: {analysis_dir}")
        
        # 读取原始数据
        if not os.path.exists(original_data_path):
            print(f"❌ 原始数据文件不存在: {original_data_path}")
            return False
            
        original_data = pd.read_csv(original_data_path)
        print(f"📊 原始数据维度: {original_data.shape}")
        
        # 查找投票结果文件
        voting_files = []
        if os.path.exists(analysis_dir):
            for file in os.listdir(analysis_dir):
                if file.startswith("features_voted_by_") and file.endswith("_methods.csv"):
                    # 提取方法数量
                    try:
                        method_count = int(file.split("_")[3])
                        voting_files.append((method_count, file))
                    except (IndexError, ValueError):
                        continue
        
        if not voting_files:
            print(f"❌ 未找到投票结果文件在: {analysis_dir}")
            return False
        
        # 按方法数量排序，选择最高投票数且特征数>=min_features的结果
        voting_files.sort(reverse=True)
        selected_features = []
        selected_method_count = 0
        
        for method_count, filename in voting_files:
            filepath = os.path.join(analysis_dir, filename)
            try:
                voting_df = pd.read_csv(filepath)
                if len(voting_df) >= min_features:
                    selected_features = voting_df['feature'].tolist()
                    selected_method_count = method_count
                    print(f"✅ 选择 {method_count} 个方法投票的结果: {len(selected_features)} 个特征")
                    break
            except Exception as e:
                print(f"⚠️ 读取文件失败 {filename}: {e}")
                continue
        
        if not selected_features:
            print(f"❌ 未找到满足最少 {min_features} 个特征的投票结果")
            return False
        
        # 准备最终的列列表
        final_columns = []
        
        # 添加ID列
        if id_col and id_col in original_data.columns:
            final_columns.append(id_col)
            print(f"📋 包含ID列: {id_col}")
        
        # 添加目标列
        if target_col and target_col in original_data.columns:
            final_columns.append(target_col)
            print(f"🎯 包含目标列: {target_col}")
        
        # 添加协变量
        if covariates:
            available_covariates = [col for col in covariates if col in original_data.columns]
            final_columns.extend(available_covariates)
            print(f"🔧 包含协变量: {available_covariates}")
        
        # 添加选中的特征
        available_features = [feat for feat in selected_features if feat in original_data.columns]
        final_columns.extend(available_features)
        
        # 去重并保持顺序
        final_columns = list(dict.fromkeys(final_columns))
        
        print(f"📊 最终数据集包含:")
        print(f"   - ID列: {1 if id_col and id_col in final_columns else 0}")
        print(f"   - 目标列: {1 if target_col and target_col in final_columns else 0}")
        print(f"   - 协变量: {len([c for c in (covariates or []) if c in final_columns])}")
        print(f"   - 筛选特征: {len(available_features)}")
        print(f"   - 总列数: {len(final_columns)}")
        
        # 创建最终数据集
        final_dataset = original_data[final_columns].copy()
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存最终数据集
        final_dataset.to_csv(output_path, index=False)
        print(f"💾 最终数据集保存到: {output_path}")
        
        # 保存特征选择报告
        report_path = output_path.replace('.csv', '_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("特征选择报告\n")
            f.write("="*50 + "\n\n")
            f.write(f"原始数据: {original_data_path}\n")
            f.write(f"原始维度: {original_data.shape}\n")
            f.write(f"最终维度: {final_dataset.shape}\n\n")
            f.write(f"选择策略: {selected_method_count} 个方法投票\n")
            f.write(f"最少特征阈值: {min_features}\n\n")
            f.write("包含的列:\n")
            if id_col and id_col in final_columns:
                f.write(f"- ID列: {id_col}\n")
            if target_col and target_col in final_columns:
                f.write(f"- 目标列: {target_col}\n")
            if covariates:
                available_covs = [c for c in covariates if c in final_columns]
                if available_covs:
                    f.write(f"- 协变量: {', '.join(available_covs)}\n")
            f.write(f"- 筛选特征 ({len(available_features)}个):\n")
            for i, feat in enumerate(available_features, 1):
                f.write(f"  {i:2d}. {feat}\n")
        
        print(f"📋 特征选择报告保存到: {report_path}")
        return True
        
    except Exception as e:
        print(f"❌ 创建最终数据集失败: {e}")
        return False


def analyze_feature_voting(results_base_dir: str = "feature_selection_results", 
                         threshold: float = 0.5,
                         output_dir: str = "feature_voting_analysis") -> Dict[str, List[str]]:
    """
    分析不同方法间的特征投票情况
    
    Args:
        results_base_dir: 特征选择结果的基础目录
        threshold: 特征被选中的频率阈值
        output_dir: 输出目录
        
    Returns:
        Dict: 不同投票数量对应的特征集合
    """
    print("\n🗳️  开始特征投票分析...")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 首先获取每个方法的稳健特征
    print("📊 获取各方法的稳健特征...")
    robust_results = analyze_robust_features(results_base_dir, threshold, 
                                           os.path.join(output_dir, "method_analysis"))
    
    if not robust_results:
        print("❌ 没有找到稳健特征结果")
        return {}
    
    # 收集所有方法的稳健特征
    method_robust_features = {}
    all_features = set()
    
    for method, results in robust_results.items():
        robust_features = set(results['robust_features'])
        method_robust_features[method] = robust_features
        all_features.update(robust_features)
        print(f"   {method}: {len(robust_features)} 个稳健特征")
    
    print(f"\n🔢 总计发现 {len(all_features)} 个唯一稳健特征")
    print(f"📊 参与投票的方法数: {len(method_robust_features)}")
    
    # 统计每个特征被多少个方法选中
    feature_votes = defaultdict(list)
    
    for feature in all_features:
        voting_methods = []
        for method, features in method_robust_features.items():
            if feature in features:
                voting_methods.append(method)
        feature_votes[len(voting_methods)].append({
            'feature': feature,
            'methods': voting_methods
        })
    
    # 按投票数量分组
    voting_results = {}
    
    print(f"\n🗳️  投票结果统计:")
    print("-" * 40)
    
    for vote_count in sorted(feature_votes.keys(), reverse=True):
        features_info = feature_votes[vote_count]
        feature_names = [info['feature'] for info in features_info]
        voting_results[f"{vote_count}_methods"] = set(feature_names)
        
        print(f"📊 {vote_count} 个方法共同选择: {len(feature_names)} 个特征")
        
        # 保存详细结果
        if features_info:
            vote_df = pd.DataFrame([
                {
                    'feature': info['feature'],
                    'vote_count': vote_count,
                    'voting_methods': ', '.join(info['methods'])
                }
                for info in features_info
            ])
            
            vote_file = os.path.join(output_dir, f"features_voted_by_{vote_count}_methods.csv")
            vote_df.to_csv(vote_file, index=False)
            print(f"   💾 保存到: {vote_file}")
            
            # 显示前几个特征作为示例
            if len(feature_names) <= 10:
                print(f"   🔍 特征: {', '.join(feature_names)}")
            else:
                print(f"   🔍 前5个特征: {', '.join(feature_names[:5])}...")
    
    # 创建投票矩阵
    print(f"\n📊 创建方法-特征投票矩阵...")
    
    methods = list(method_robust_features.keys())
    features = sorted(all_features)
    
    # 创建投票矩阵
    vote_matrix = []
    for feature in features:
        row = {'feature': feature}
        vote_count = 0
        for method in methods:
            voted = feature in method_robust_features[method]
            row[method] = 1 if voted else 0
            if voted:
                vote_count += 1
        row['total_votes'] = vote_count
        vote_matrix.append(row)
    
    # 保存投票矩阵
    matrix_df = pd.DataFrame(vote_matrix)
    matrix_df = matrix_df.sort_values('total_votes', ascending=False)
    matrix_file = os.path.join(output_dir, "feature_voting_matrix.csv")
    matrix_df.to_csv(matrix_file, index=False)
    print(f"💾 投票矩阵保存到: {matrix_file}")
    
    # 创建汇总统计
    summary_stats = []
    total_methods = len(methods)
    
    for vote_count in range(1, total_methods + 1):
        if vote_count in feature_votes:
            count = len(feature_votes[vote_count])
            percentage = (count / len(all_features)) * 100 if all_features else 0
            summary_stats.append({
                'vote_count': vote_count,
                'feature_count': count,
                'percentage': percentage,
                'description': f"{vote_count}/{total_methods} 方法一致"
            })
    
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        summary_file = os.path.join(output_dir, "voting_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"💾 投票汇总保存到: {summary_file}")
    
    # 特别关注的结果
    print(f"\n🎯 关键发现:")
    print("-" * 30)
    
    if total_methods >= 2 and 2 in feature_votes:
        print(f"🤝 两个方法共同选择: {len(feature_votes[2])} 个特征")
    
    if total_methods >= 3 and 3 in feature_votes:
        print(f"🤝 三个方法共同选择: {len(feature_votes[3])} 个特征")
        
    if total_methods >= 4 and 4 in feature_votes:
        print(f"🤝 四个方法共同选择: {len(feature_votes[4])} 个特征")
    
    if total_methods in feature_votes:
        consensus_features = len(feature_votes[total_methods])
        print(f"🎯 所有方法一致选择: {consensus_features} 个特征")
        if consensus_features > 0:
            print("   这些是最稳健的特征!")
    
    print(f"\n✅ 特征投票分析完成! 结果保存在: {output_dir}")
    return voting_results

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Feature Selection Pipeline")
    parser.add_argument("--data", required=True,
                       help="Input data file path")
    parser.add_argument("--target-col", required=True,
                       help="Target column name")
    
    # 不平衡数据集流程参数
    parser.add_argument("--iterations", type=int,
                       help="Number of iterations")
    parser.add_argument("--match-cols", nargs='+',
                       help="Columns to use for matching (space-separated)")
    parser.add_argument("--match-ratio", type=int, default=3,
                       help="Match ratio (default: 3)")
    parser.add_argument("--random-state", type=int, default=123,
                       help="Random state (default: 123)")
    
    # 特征选择相关参数
    parser.add_argument("--feature-methods", nargs='+', 
                       default=['randomforest', 'elasticnet'],
                       help="Feature selection methods to use")
    parser.add_argument("--top-k", type=int, default=50,
                       help="Number of top features to select (default: 50)")
    parser.add_argument("--id-col", default=None,
                       help="ID column name (optional)")
    
    # 特征分析相关参数
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Threshold for robust feature selection (default: 0.5)")
    parser.add_argument("--analysis-output", default="feature_analysis",
                       help="Output directory for analysis results")
    
    # 数据集平衡性相关参数
    parser.add_argument("--force-strategy", choices=['balanced', 'imbalanced', 'auto'],
                       default='auto', help="Force specific strategy (default: auto)")
    
    # 最终数据集生成参数
    parser.add_argument("--min-features", type=int, default=10,
                       help="Minimum number of features for final dataset (default: 10)")
    parser.add_argument("--covariates", nargs='+', 
                       help="Covariates to keep in final dataset")
    
    args = parser.parse_args()
    
    # 首先分析数据集平衡性
    if not args.data or not args.target_col:
        parser.error("数据集分析需要 --data 和 --target-col 参数")
    
    # 分析数据集平衡性（使用固定阈值0.3）
    balance_info = analyze_dataset_balance(args.data, args.target_col, balance_threshold=0.3)
    
    # 决定使用哪种策略
    if args.force_strategy == 'auto':
        strategy = balance_info['recommended_strategy']
    else:
        strategy = args.force_strategy
        print(f"\n🔧 用户强制指定策略: {strategy}")
    
    print(f"\n🎯 采用策略: {strategy.upper()}")
    
    if strategy == 'balanced':
        # 平衡数据集策略：直接进行特征选择
        print("\n" + "="*60)
        print("🎯 执行平衡数据集特征选择流程")
        
        feature_results = run_balanced_feature_selection(
            data_path=args.data,
            target_col=args.target_col,
            methods=args.feature_methods,
            top_k=args.top_k,
            output_dir="feature_selection_outputs/feature_selection",
            id_col=args.id_col
        )
        
        # 对平衡数据集的结果进行投票分析
        if feature_results:
            print("\n" + "="*60)
            print("🗳️ 分析特征选择结果的交集...")
            
            # 创建临时结果目录结构供投票分析使用
            import shutil
            temp_results_dir = "temp_balanced_results"
            os.makedirs(temp_results_dir, exist_ok=True)
            
            # 复制结果文件到临时目录
            for method, result_file in feature_results.items():
                temp_method_dir = os.path.join(temp_results_dir, f"res_{method}_1")
                os.makedirs(temp_method_dir, exist_ok=True)
                
                # 复制文件
                shutil.copy2(result_file, os.path.join(temp_method_dir, f"{method}_test_1.csv"))
            
            # 进行投票分析（阈值设为1.0，因为只有一次选择）
            voting_results = analyze_feature_voting(
                results_base_dir=temp_results_dir,
                threshold=1.0,  # 平衡数据集只做一次选择，所以阈值设为1.0
                output_dir=get_feature_selection_subdir("analysis")
            )
            
            # 清理临时目录
            shutil.rmtree(temp_results_dir)
            
            print(f"\n🎉 平衡数据集分析完成!")
            print(f"📁 特征选择结果: {get_feature_selection_subdir('feature_selection')}/")
            print(f"📊 投票分析结果: {get_feature_selection_subdir('analysis')}/")
            
            # 创建最终数据集
            final_success = create_final_feature_dataset(
                original_data_path=args.data,
                analysis_dir=get_feature_selection_subdir("analysis"),
                output_path=os.path.join(get_feature_selection_dir(), "feature_selected_data.csv"),
                min_features=args.min_features,
                covariates=args.covariates,
                target_col=args.target_col,
                id_col=args.id_col
            )
            if final_success:
                print(f"🎉 最终数据集创建成功!")
        
        exit(0)
    
    else:
        # 不平衡数据集策略：使用原有的重采样流程
        print("\n" + "="*60)
        print("⚖️ 执行不平衡数据集特征选择流程（包含重采样）")
        
        # 验证不平衡数据集流程的必需参数
        required_args = [ 'iterations', 'match_cols']
        missing_args = [arg for arg in required_args if getattr(args, arg.replace('-', '_')) is None]
        if missing_args:
            parser.error(f"不平衡数据集流程需要以下参数: {', '.join(['--' + arg for arg in missing_args])}")
        
        # 运行数据匹配管道（固定75%采样比例）
        success = run_feature_selection_pipeline(
            data_path=args.data,
            n_iterations=args.iterations,
            target_col=args.target_col,
            match_cols=args.match_cols,
            match_ratio=args.match_ratio,
            sample_ratio=0.75,  # 固定75%
            random_state=args.random_state
        )
        
        if success:
            print(f"\n✅ Pipeline completed successfully!")
            print(f"Generated {args.iterations} matched datasets")
            print(f"Output directory: {get_feature_selection_subdir('resampling')}")
            
            # 运行特征选择方法
            feature_results = run_feature_selection_methods(
                input_dir=get_feature_selection_subdir("resampling"),
                target_col=args.target_col,
                iterations=args.iterations,
                methods=args.feature_methods,
                top_k=args.top_k,
                output_base_dir=get_feature_selection_subdir("feature_selection"),
                id_col=args.id_col
            )
            
            print("\n✅ 特征选择完成!")
            for method, stats in feature_results.items():
                print(f"  {method}: {stats['completed_files']} 文件 -> {stats['output_dir']}")
            
            # 自动运行特征分析
            print("\n" + "="*60)
            print("🔍 开始自动特征分析...")
            
            print("\n📊 步骤 1: 稳健特征分析...")
            robust_results = analyze_robust_features(
                results_base_dir=get_feature_selection_subdir("feature_selection"),
                threshold=args.threshold,
                output_dir=os.path.join(get_feature_selection_subdir("analysis"), "robust_features")
            )
            
            print("\n🗳️ 步骤 2: 特征投票分析...")
            voting_results = analyze_feature_voting(
                results_base_dir=get_feature_selection_subdir("feature_selection"),
                threshold=args.threshold,
                output_dir=get_feature_selection_subdir("analysis")
            )
            
            print(f"\n🎉 完整流程完成!")
            print(f"📁 特征选择结果: {get_feature_selection_subdir('feature_selection')}/")
            print(f"📊 分析结果: {get_feature_selection_subdir('analysis')}/")
            
            # 显示关键发现
            if voting_results and '6_methods' in voting_results:
                consensus_count = len(voting_results['6_methods'])
                if consensus_count > 0:
                    print(f"🎯 发现 {consensus_count} 个所有方法一致选择的最稳健特征!")
            
            # 创建最终数据集
            final_success = create_final_feature_dataset(
                original_data_path=args.data,
                analysis_dir=get_feature_selection_subdir("analysis"),
                output_path=os.path.join(get_feature_selection_dir(), "feature_selected_data.csv"),
                min_features=args.min_features,
                covariates=args.covariates,
                target_col=args.target_col,
                id_col=args.id_col
            )
            if final_success:
                print(f"🎉 最终数据集创建成功!")
        
        exit(0 if success else 1)

