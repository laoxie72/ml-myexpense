import pandas as pd
from sqlalchemy import create_engine
import yaml
import os

def load_config(config_path: str) -> dict:
    """
    加载 YAML 配置文件。

    Args:
        config_path (str): 配置文件路径。

    Returns:
        dict: 配置内容。
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_connection_string(config: dict) -> str:
    """
    构造数据库连接字符串。

    Args:
        config (dict): 数据库配置信息。

    Returns:
        str: 数据库连接字符串。
    """
    user = config['user']
    password = config['password']
    host = config['host']
    port = config['port']
    db_name = config['db_name']
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}"

def fetch_data(query: str, config_path: str) -> pd.DataFrame:
    """
    从数据库中提取数据。

    Args:
        query (str): SQL 查询语句。
        config_path (str): 配置文件路径。

    Returns:
        pd.DataFrame: 查询结果的数据框。
    """
    # 加载配置
    config = load_config(config_path)
    # 构造连接字符串
    connection_string = get_connection_string(config['database'])
    # 连接数据库并执行查询
    engine = create_engine(connection_string, pool_pre_ping=True)
    with engine.connect() as conn:
        data = pd.read_sql(query, conn)
    return data
