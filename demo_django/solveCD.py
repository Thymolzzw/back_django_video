
from django.middleware.security import SecurityMiddleware
from django.utils.deprecation import MiddlewareMixin
class MyCorsMiddle(MiddlewareMixin):
    def process_response(self, request, response):
        # if request.method == 'OPTIONS':
            # 允许Content-Type类型
        response['Access-Control-Allow-Headers'] = '*'
        # response['Access-Control-Allow-Credentials'] = 'true'
        # response['Accept-Ranges'] = 'bytes'
        # response['Allow'] = '*'
            # 允许所有的header
            # obj['Access-Control-Allow-Headers']='*'
            # 允许某个ip+port
            # obj['Access-Control-Allow-Origin']='http://127.0.0.1:8000'
        # response['Access-Control-Allow-Origin'] = 'http://localhost:9527'
        response['Access-Control-Allow-Origin'] = '*'
        return response