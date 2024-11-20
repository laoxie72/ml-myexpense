import pymysql
import pandas as pd

def read_data(query, db_config):
    """
    从数据库中读取数据。
    :param query: SQL 查询语句
    :param db_config: 数据库配置字典，包含 host, user, password, database 等
    :return: Pandas DataFrame
    """
    connection = pymysql.connect(**db_config)
    try:
        df = pd.read_sql(query, connection)
    finally:
        connection.close()
    return df

def write_data(df, table_name, db_config):
    """
    将数据写入数据库。
    :param df: 待写入的 DataFrame
    :param table_name: 数据库表名
    :param db_config: 数据库配置字典
    """
    connection = pymysql.connect(**db_config)
    try:
        df.to_sql(table_name, connection, if_exists="replace", index=False)
    finally:
        connection.close()
