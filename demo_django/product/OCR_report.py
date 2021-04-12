import json
import os
import pdfkit


def test():
    # content = '这是一个测试文件。' + '<br>' + 'Hello from Python!'
    content =  'output\nuser\ninput\n(p 2.93)\nLearning\nfunction\na violation\nOARPA\nlearned\nWhy did you do that?\nWhy not something else?\nWhen do you fail\nHow is it done today?\nWhen do you succee,\ntraining data\nprocess\nWhen can l trust you?\nThis incident is\nAI COLLOOUUM\nHow do I correct an error?\nDetecting ceasefire violations\nwien 68%0016\n'
    html = '<html><head><meta charset="UTF-8"></head><body><div align="center"><p>%s</p></div></body></html>'%content
                            
    pdfkit.from_string(html, 'test.pdf')
    

def ocr_report_return_with_path(text_location_path):
    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = 'demo_django'
    curPath = curPath.split(split_reg)[0] + split_reg
    print("text_location_path", text_location_path)
    try:
        with open(text_location_path, 'r') as load_f:
            text_data = json.load(load_f)
        # print("text_data", text_data)
        all_content = ''
        # if os.path.exists(os.path.join(curPath, 'demo_django', 'product', 'test.html')):
            # print("chunzai")
        with open(os.path.join(curPath, 'demo_django', 'product', 'test.html'), encoding = 'UTF-8') as file_obj:
            # print("jinru")
            content = file_obj.read()
            # print("content", content)
        for i in range(len(text_data)):
            # print(text_data[i])
            content_i = content % (curPath + '/statics/resource/text_images/' + text_data[i]["image"], text_data[i]["time"], text_data[i]["content"])
            all_content += content_i
        html = '<html><head><meta charset="UTF-8"></head><body>%s</body></html>'%(all_content)
        # print(html)
        text_pdf_db = 'statics/resource/text_pdf/' + text_location_path.split('/')[-1].split('.')[0] + '.pdf'
        test_pdf_path = os.path.join(curPath, text_pdf_db)
        optionsss = {
            'enable-local-file-access': '--enable-local-file-access'
        }
        pdfkit.from_string(html, test_pdf_path, options=optionsss)
        return text_pdf_db
    except Exception:
        # print(Exception)
        pass

if __name__ == '__main__':
    text_location_path = '/home/wh/zzw/demo_django/statics/resource/text/mytest_4ca13bc2-5e14-11eb-b8f4-b0a460e7b2fb.json'
    output_path = ocr_report_return_with_path(text_location_path)
    print(output_path)
    # test()