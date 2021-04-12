import hashlib


def getBaseUrl():
    return "http://127.0.0.1:9000/"
    # return "http://cuiky.natapp1.cc/"

def md5Encode(str):
    m = hashlib.md5()
    b = str.encode(encoding='utf-8')
    m.update(b)
    str_md5 = m.hexdigest()
    return str_md5

if __name__ == '__main__':
    print(md5Encode('aaaaaa'))