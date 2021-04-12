import pymysql
import numpy as np
#读取数据库中的声纹特征
def sql_find():
    # 连接数据库
    count = pymysql.connect(
        host='127.0.0.1',  # 数据库地址
        port=3300,  # 数据库端口号
        user='root',  # 数据库账号
        password='123456',  # 数据库密码
        db='graduate_test')  # 数据库名称
    # 创建数据库对象
    db = count.cursor()
    # 写入SQL语句
    sql = "select * from voice_feature"
    try:
        # 执行sql
        db.execute(sql)
        # 提交事务
        result=db.fetchall()
        return result
    except Exception as e:
        print(e)
        count.rollback()
        print('加载特征失败')
    finally:
        count.close()
    db.close()

r=sql_find()
for i in r:
    a=np.load(i[0])
    print(a)
