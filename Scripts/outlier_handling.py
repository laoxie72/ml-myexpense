import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_outliers_with_isolation_forest(data: pd.DataFrame, feature_columns: list, contamination: float = 0.05) -> pd.DataFrame:
    """
    使用 Isolation Forest 检测并标记异常值。

    Args:
        data (pd.DataFrame): 输入数据。
        feature_columns (list): 要检测异常值的特征列。
        contamination (float): 异常点比例，默认为 0.05。

    Returns:
        pd.DataFrame: 标记了异常值的数据，新增一列 'is_outlier' 表示是否为异常点（1 表示异常）。
    """
    # 检查特征列是否存在于数据中
    for col in feature_columns:
        if col not in data.columns:
            raise ValueError(f"特征列 {col} 不在数据中。")

    # 初始化 Isolation Forest 模型
    model = IsolationForest(contamination=contamination, random_state=42)

    # 训练模型并标记异常值
    data['is_outlier'] = model.fit_predict(data[feature_columns])

    # 将标记结果转换为 0（正常）和 1（异常）
    data['is_outlier'] = data['is_outlier'].apply(lambda x: 1 if x == -1 else 0)

    return data

def remove_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """
    删除标记为异常值的数据。

    Args:
        data (pd.DataFrame): 包含异常值标记的数据，需包含 'is_outlier' 列。

    Returns:
        pd.DataFrame: 删除异常值后的数据。
    """
    if 'is_outlier' not in data.columns:
        raise ValueError("数据中没有 'is_outlier' 列，无法删除异常值。")
    return data[data['is_outlier'] == 0].drop(columns=['is_outlier'])