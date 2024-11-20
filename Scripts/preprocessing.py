import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize_features(data: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    对指定的特征列进行归一化（Min-Max Scaling）。

    Args:
        data (pd.DataFrame): 原始数据。
        feature_columns (list): 需要归一化的列名列表。

    Returns:
        pd.DataFrame: 包含归一化结果的数据。
    """
    # 检查特征列是否存在于数据中
    for col in feature_columns:
        if col not in data.columns:
            raise ValueError(f"特征列 {col} 不在数据中。")

    # 初始化 MinMaxScaler
    scaler = MinMaxScaler()

    # 对指定列进行归一化
    data[feature_columns] = scaler.fit_transform(data[feature_columns])

    return data
