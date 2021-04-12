import hashlib
import random
import requests

# set baidu develop parameter
apiurl = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
appid = '20210124000679939'
secretKey = '1azBAujPnysgE8YBhz5R'


# 翻译内容 源语言 翻译后的语言
def translateBaidu(content, fromLang='en', toLang='zh'):
    salt = str(random.randint(32768, 65536))
    sign = appid + content + salt + secretKey
    sign = hashlib.md5(sign.encode("utf-8")).hexdigest()
    try:
        paramas = {
            'appid': appid,
            'q': content,
            'from': fromLang,
            'to': toLang,
            'salt': salt,
            'sign': sign
        }
        response = requests.get(apiurl, paramas)
        jsonResponse = response.json()  # 获得返回的结果，结果为json格式
        dst = str(jsonResponse["trans_result"][0]["dst"])  # 取得翻译后的文本结果
        return dst
    except Exception as e:
        print(e)

def translate_api(content):
    return translateBaidu(content)


if __name__ == '__main__':
    str1 = 'We hold these truths will be self-evident, that all men were created equal'
    result = translateBaidu(str1)
    print(result)