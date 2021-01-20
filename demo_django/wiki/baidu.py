import urllib.request
import urllib.parse
from lxml import etree
import csv
def query(content):
    # 请求地址
    url = 'https://baike.baidu.com/item/' + urllib.parse.quote(content)

    print("url", url)

    # 请求头部
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
    }
    # 利用请求地址和请求头部构造请求对象
    req = urllib.request.Request(url=url, headers=headers, method='GET')
    # 发送请求，获得响应
    response = urllib.request.urlopen(req)
    # 读取响应，获得文本
    text = response.read().decode('utf-8')

    print("text:", text)

    # 构造 _Element 对象
    html = etree.HTML(text)
    # 使用 xpath 匹配数据，得到匹配字符串列表
    sen_list = html.xpath('//div[contains(@class,"lemma-summary") or contains(@class,"lemmaWgt-lemmaSummary")]//text()')
    # 过滤数据，去掉空白
    sen_list_after_filter = [item.strip('\n') for item in sen_list]
    # 将字符串列表连成字符串并返回
    #return ''.join(sen_list_after_filter)
    result=''.join(sen_list_after_filter)
    # 1. 创建文件对象
    f = open('data.csv', 'w', encoding='utf-8', newline='')

    # 2. 基于文件对象构建 csv写入对象
    csv_writer = csv.writer(f)

    # 3. 构建列表头
    # csv_writer.writerow(["姓名", "年龄", "性别"])

    # 4. 写入csv文件内容
    csv_writer.writerow([content, result])

    # 5. 关闭文件
    f.close()
if __name__ == '__main__':
    # content = input('查询词语：')
    content = '中国'
    result = query(content)
    print("查询结果：%s" % result)


# if __name__ == '__main__':
#     while (True):
#         content = input('查询词语：')
#         result = query(content)
#         print("查询结果：%s" % result)
#         # 1. 创建文件对象
#         f = open('introduce/data.csv', 'a', encoding='utf-8',newline='')
#
#         # 2. 基于文件对象构建 csv写入对象
#         csv_writer = csv.writer(f)
#
#         # 3. 构建列表头
#         # csv_writer.writerow(["姓名", "年龄", "性别"])
#
#         # 4. 写入csv文件内容
#         csv_writer.writerow([content,result])
#
#
#         # 5. 关闭文件
#         f.close()