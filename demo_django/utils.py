import hashlib


def getBaseUrl():
    # return "http://172.26.124.167:8000/"
    return "http://videos.natapp1.cc/"

def md5Encode(str):
    m = hashlib.md5()
    b = str.encode(encoding='utf-8')
    m.update(b)
    str_md5 = m.hexdigest()
    return str_md5
