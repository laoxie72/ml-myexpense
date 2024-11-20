import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from Scripts.data_cleaning import clean_data
from Scripts.data_pipeline import fetch_data
from Scripts.outlier_handling import detect_outliers_with_isolation_forest, remove_outliers
from Scripts.preprocessing import normalize_features


if __name__ == "__main__":
    # 配置文件路径
    config_path = "config/db_config.yaml"
    # SQL 查询
    query = "SELECT id,`金额（可执行）` FROM `table_detail` WHERE `项目` != '小结' AND `金额（可执行）` < 0;"

    # 获取数据
    raw_data = fetch_data(query, config_path)
    # print(data.head())
    # 原始数据
    # print(raw_data.describe())

    # 数据清洗
    cleaned_data = clean_data(raw_data)
    # print("清洗后的数据:")
    # print(cleaned_data.head())

    # 指定需要归一化的列
    feature_columns = ["id","金额（可执行）"]  # 替换为实际列名
    normalized_data = normalize_features(cleaned_data, feature_columns)
    # print("归一化后的数据:")
    # print(normalized_data.head())

    # 检测异常值
    processed_data = detect_outliers_with_isolation_forest(normalized_data, feature_columns)

    # 打印结果
    # print("标记异常值后的数据:")
    # print(processed_data.head())

    # 分离正常点和异常点
    normal_data = processed_data[processed_data['is_outlier'] == 0]
    outlier_data = processed_data[processed_data['is_outlier'] == 1]
    # print("正常数据:")
    # print(normal_data.head())
    # print("异常数据:")
    # print(outlier_data.head())

    # 检测异常值
    data_with_outliers = detect_outliers_with_isolation_forest(normalized_data, feature_columns)

    # 删除异常值
    data_without_outliers = remove_outliers(data_with_outliers)
    # print("删除异常值后的数据:")
    # print(data_without_outliers.head())

    # 输入特征和目标变量
    X = data_without_outliers["id"]  # 替换为实际的特征列名
    y = data_without_outliers['金额（可执行）']  # 替换为目标变量的列名

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 如果 X_train 是 Series，将其转换为 DataFrame
    if isinstance(X_train, pd.Series):
        X_train = X_train.to_frame()
    # 如果 X_test 是 Series，将其转换为 DataFrame
    if isinstance(X_test, pd.Series):
        X_test = X_test.to_frame()

    # print(f"训练集大小: {X_train.shape}")
    # print(f"测试集大小: {X_test.shape}")
    #
    # print("X_train 类型:", type(X_train))
    # print("X_train 形状:", X_train.shape)
    # print("y_train 类型:", type(y_train))
    # print("y_train 形状:", y_train.shape)

    # 初始化随机森林回归模型
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # 模型训练
    rf_model.fit(X_train, y_train)

    print("随机森林模型训练完成！")

    # 测试集预测
    y_pred = rf_model.predict(X_test)

    # 评估指标计算
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"均方误差 (MSE): {mse}")
    print(f"决定系数 (R²): {r2}")

    # 绘制实际值 vs. 预测值
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', label="理想预测线")
    plt.xlabel("实际值")
    plt.ylabel("预测值")
    plt.title("随机森林回归 - 实际值 vs. 预测值")
    plt.legend()
    # plt.show()

    # 保存模型
    joblib.dump(rf_model, "models/random_forest_model.pkl")
    print("随机森林模型已保存到 models/random_forest_model.pkl")

    # 加载模型
    loaded_model = joblib.load("models/random_forest_model.pkl")
    print("模型加载成功！")

    # 使用加载的模型进行预测
    new_predictions = loaded_model.predict(X_test)
    print(f"加载模型后的预测结果: {new_predictions[:5]}")

    # 初始化 MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # 归一化目标变量
    y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))

    # 训练模型
    rf_model.fit(X_train, y_train_scaled)

    # 做出预测（归一化后的预测结果）
    y_pred_scaled = rf_model.predict(X_test)

    # 反归一化预测结果
    y_pred_original = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

    # 输出反归一化后的预测结果
    print(y_pred_original)


