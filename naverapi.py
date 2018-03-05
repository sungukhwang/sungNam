# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 21:35:04 2018

@author: pc
"""

# -*- coding: utf-8 -*-
#!/usr/bin/python
# -*- coding: cp949 -*-


"""
Created on Thu Jan 18 09:50:56 2018

@author: pc
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 12:01:51 2018

@author: pc
"""

# --*-- coding: utf-8 --*--
import codecs
import re
import collections
import requests
import time
from bs4 import BeautifulSoup
import string
import urllib
import urllib.request
import urllib.parse
import webbrowser
# 에러 리스트 생성 함수
def insert_error(error, error_doc):
    for i in error_doc:
        error_log = str(error_doc[i]) + " / " + str(error_doc["page"]) + "page / " + str(error_doc["post_number"]) \
                    + "th post / " + error_doc["title"] \
                    + " / http://blog.naver.com/PostList.nhn?blogId=happy_inhatc&currentPage=" + str(
            error_doc["page"])
    error_list.append(error_log)


total_num = 0;

error_list = []

print("블로그 ID->")
blog_id = input()

print("\n탐색 시작 페이지 수->")
start_p = int(input())

print("\n탐색 종료 페이지 수->")
end_p = int(input())

print("\nCreating File Naver_Blog_Crawling_Result.txt...\n")

# 파일 열기
file = codecs.open("Naver_Blog_Crawling_Result1.txt", 'w', encoding="utf-8")

# 페이지 단위
for page in range(start_p, end_p + 1):
    doc = collections.OrderedDict()

    url = "http://blog.naver.com/PostList.nhn?blogId=" + blog_id + "&currentPage=" + str(page)
    r = requests.get(url)
    if (not r.ok):
        print("Page" + page + "연결 실패, Skip")
        continue

    # html 파싱
    soup = BeautifulSoup(r.text.encode("utf-8"), "html.parser")

    # 페이지 당 포스트 수 (printPost_# 형식의 id를 가진 태그 수)
    post_count = len(soup.find_all("table", {"id": re.compile("printPost.")}))

    doc["page"] = page

    # 포스트 단위
    for pidx in range(1, post_count + 1):
        doc["post_number"] = pidx
        post = soup.find("table", {"id": "printPost" + str(pidx)})


        # 제목 찾기---------------------------

        title = post.find("h3", {"class": "se_textarea"})

        # 스마트에디터3 타이틀 제거 임시 적용 (클래스가 다름)
        if (title == None):
            title = post.find("span", {"class": "pcol1 itemSubjectBoldfont"})

        if (title != None):
            doc["title"] = title.text.strip()
        else:
            doc["title"] = "TITLE ERROR"

        # 날짜 찾기---------------------------

        date = post.find("span", {"class": "se_publishDate pcol2 fil5"})

        # 스마트에디터3 타이틀 제거 임시 적용 (클래스가 다름)
        if (date == None):
            date = post.find("p", {"class": "date fil5 pcol2 _postAddDate"})

        if (date != None):
            doc["date"] = date.text.strip()
        else:
            doc["date"] = "DATE ERROR"


        # 내용 찾기---------------------------

        content = post.find("div", {"class": "se_component_wrap sect_dsc __se_component_area"})

        # 스마트에디터3 타이틀 제거 임시 적용 (클래스가 다름)
        if (content == None):
            content = post.find("div", {"id": "postViewArea"})

        if (title != None):
            # Enter 5줄은 하나로
            doc["며여용"] =content.text.strip().replace("\n" * 5, "\n")
        else:
            doc["며여용"] = "CONTENT ERROR"

        # doc 출력 (UnicodeError - 커맨드 창에서 실행 시 발생)
        for i in doc:

            str_doc = str(i) + ": " + str(doc[i])
            try:
                print(str_doc)
            except UnicodeError:
                print(str_doc.encode("utf-8"))

            file.write(str_doc + "\n")

            # 파일 쓰기
            if ("ERROR" in str(doc[i])):
                insert_error(doc[i], doc)

        #전체 수 증가
        total_num += 1

# 결과 출력 (전체 글 수, 에러 수)


print("Total : " + str(total_num))

error_count = len(error_list)
print("Error : " + str(error_count))

# 에러가 있을 경우 출력
if (error_count != 0):
    print("Error Post : ")
    for i in error_list:
        print(i)
file.close()     

file1=open('Naver_Blog_Crawling_Result1.txt','r', encoding="utf-8")
lines= [] 
x=[]
count=0
for paragraph in file1:
    for char in '.,!?[]@():':
        paragraph=paragraph.replace(char,'')
    lines = str.split(paragraph, "\n")
    print(lines)
    for each_line in lines:
        if each_line.find("며여용")<0:
            count+=1
        else:
            client_id="ebTOYs9YiY0RIEV8Vn8u"
            client_secret="gSw_0r8zLc"
            encText=urllib.parse.quote(each_line)
            print(each_line)
            print(encText)
            data="speaker=mijin&speed=0&text="+encText;
            url="https://openapi.naver.com/v1/voice/tts.bin"
            request=urllib.request.Request(url)
            request.add_header("X-Naver-Client-Id",client_id)
            request.add_header("X-Naver-Client-Secret",client_secret)
            response = urllib.request.urlopen(request,data=data.encode('utf-8'))
            rescode = response.getcode()
            if (rescode==200):
                print("TTS mp3 저장")
                response_body = response.read()
                for i in range(total_num):
                   with open('C:\\Users\\pc\\Desktop\\isthe\\'+str(i)+'.mp3', 'wb') as f:
                      f.write(response_body)
            else:
                print("Error Code:" + rescode)
file1.close()
 


