from lxml import etree
import urllib.request
import urllib.parse



def query(content):
    # 请求地址
    url = 'https://en.wikipedia.org/wiki/' + urllib.parse.quote(content)


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

    # print("text:", text)

    # 构造 _Element 对象
    html = etree.HTML(text)
    # 使用 xpath 匹配数据，得到 <div class="mw-parser-output"> 下所有的子节点对象
    obj_list = html.xpath('//div[@class="mw-parser-output"]/*')

    # print("ouput", obj_list)

    # # 在所有的子节点对象中获取有用的 <p> 节点对象
    # for i in range(0,len(obj_list)):
    #     if 'p' == obj_list[i].tag:
    #         start = i
    #         break
    # print("开始：", i)
    # for i in range(start,len(obj_list)):
    #     if 'p' != obj_list[i].tag:
    #         end = i
    #         break
    # print("结束：", i)

    p_list = []
    for i in range(0, len(obj_list)):
        if 'p' == obj_list[i].tag:
            p_list.append(obj_list[i])
            if len(p_list) > 2:
                break

    # 使用 xpath 匹配数据，得到 <p> 下所有的文本节点对象
    sen_list_list = [obj.xpath('.//text()') for obj in p_list]
    # 将文本节点对象转化为字符串列表
    sen_list = [sen.encode('utf-8').decode() for sen_list in sen_list_list for sen in sen_list]
    # 过滤数据，去掉空白
    sen_list_after_filter = [item.strip('\n') for item in sen_list]
    # 将字符串列表连成字符串并返回
    return ''.join(sen_list_after_filter)

    # sen = p_list[0].xpath('.//text()')
    # return sen


def wiki_api(name):
    name.strip()
    if ' ' in name:
        print("ok")
        name = name.replace(' ', '_')
    print(name)
    result = query(name)
    return result


if __name__ == '__main__':
    # content = input('Word: ')
    content = 'china'
    result = wiki_api(content)
    print("Result: %s" % result)