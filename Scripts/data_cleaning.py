import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体为 SimHei（黑体）
rcParams['font.sans-serif'] = ['SimHei']

# 避免负号显示问题
rcParams['axes.unicode_minus'] = False


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    清洗数据，包括处理缺失值、重复值和异常值。

    Args:
        data (pd.DataFrame): 原始数据。

    Returns:
        pd.DataFrame: 清洗后的数据。
    """
    # 检查数据基本信息
    # print("basic info of these data")
    # print(data.info())

    # 处理缺失值：删除或填充
    # data.dropna(inplace=True)  # 示例：直接删除缺失值
    # 或者填充缺失值
    # data.fillna(data.mean(), inplace=True)

    # 去除重复值
    data.drop_duplicates(inplace=True)

    # 检查并可视化异常值（示例：使用箱线图）
    for column in data.select_dtypes(include=['float64', 'int64']).columns:
        sns.boxplot(data[column])
        plt.title(f'{column} - Boxplot')
        # plt.show()

    # print("清洗后的数据概况:")
    # print(data.describe())
    return data
