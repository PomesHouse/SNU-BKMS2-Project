'''
"user" -> ex) "U01QNJ7PWRX"
"text" -> ex) ""\u003c!channel\u003e 제 1회..."  -> 텍스트 전처리 및 벡터화 
              "*\u0026lt;2021년 하반기 세계은행(Wor
"ts" -> ex) "1615453048.002500"
"thread_ts"
"reply_count" ->  ex) 4
"blocks" -> ex) [
      {
        "block_id": "0rad",
        "elements": [
          {
            "type": "rich_text_section",
            "elements": [
              {
                "type": "broadcast",
                "range": "channel"
              },
              {
                "type": "text",
                "text": " 제 1회 SNU GSDS InnoJam 관련 공지 (필독)\n\n2020년 12월에 2년차 학생들이 연구진행상황을 공유하는 제 1회 InnoJam 이벤트에 대해서 공지를 드렸었는데요. 당초 계획보다 다소 미뤄진 3월 26일 오후 3시30분에 이벤트를 개최하려 합니다.\n\n구성원들의 적극적인 참여를 통해 코로나로 힘든 시기에 좋은 상호 교류의 장이 마련되기를 바랍니다.\n\nInnoJam에 대한 세부사항들은 아래와 같으며, 관련 문의사항은 조직위원회에 연락 주시면 성심성의껏 답변해 드리겠습니다 ("
              },
              {
                "type": "link",
                "url": "mailto:hyungkim@snu.ac.kr",
                "text": "hyungkim@snu.ac.kr"
              },
              {
                "type": "text",
                "text": ", "
              },
              {
                "type": "link",
                "url": "mailto:minoh@snu.ac.kr",
                "text": "minoh@snu.ac.kr"
              },
              {
                "type": "text",
                "text": ").\n\n제 1회 SNU GSDS InnoJam 스케줄\n3월 4일 (목) 오후 6시: 포스터 및 소개영상 제출 시작\n3월 13일 (토) 오후 6시: 포스터 및 소개영상 제출 마감\n3월 14일 (일) 오후 6시: 투표 시작 (각 교수/학생이 자신이 생각하는 우수 포스터 최대 5편 선정. 1~5등 순위는 없음)\n3월 19일 (금) 오후 1시30분: 소개영상 전체재생\n3월 21일 (일) 오후 6시: 투표 종료\n3월 22일 (월) 오후 6시: 투표 결과 발표 (Top 8 포스터 선정)\n3월 26일 (금) 오후 3시30분 – 6시: Top 8 연구의 구두 발표, 교수 채점 및 Top 3 시상식\n전체 포스터들은 대학원 곳곳에 일정 기간 동안 전시 예정\n\n관련 규정\n모든 2년차 학생(2020년 3월 입학생)은 적어도 하나의 (1) A0 크기 포스터와 (2) 90초이내의 소개 영상을 제출해야 함 (개인 혹은 그룹 포스터 모두 인정, 포스터 및 영상은 영어로 작성)\n각 교수/학생은 최대 5편의 우수 포스터를 선택 가능 (1~5등 순위 없이 같은 점수 부여)\n개인/그룹 프로젝트 모두 동일한 방식으로 투표\n교수 투표 결과와 학생 투표 결과를 50%씩 반영하여 최고점을 받은 8편을 구두 발표 대상으로 선정\n선정된 연구는 3월 26일 InnoJam 이벤트에서 구두 발표 (연구 별로 10분 발표 및 2분 질의응답, 발표는 영어로 진행)\n구두 발표 8건 중 교수 채점을 통해 Top 3를 선정하여 시상\n\n\n\n\n제출/투표 가이드\n"
              },
              {
                "type": "link",
                "url": "https://docs.google.com/document/d/1RGBAvYiFexcxg1ACw1h-fgxP0swpHrlF8Q3W4Di001U/edit?usp=sharing",
                "text": ""
              },
              {
                "type": "text",
                "text": "\n\nGSDS InnoJam 조직위원회 (Wen-Syan Li, 김형신, 오민환)\n\n※ 참고 가능한 샘플을 첨부하니 확인하시기 바랍니다.\n\n"
              },
              {
                "type": "link",
                "url": "https://drive.google.com/file/d/1WigdZA7iMMf5u8QsAcvizvwFB-7rlH8n/view?usp=sharing",
                "text": ""
              },
              {
                "type": "text",
                "text": "\n\n"
              },
              {
                "type": "link",
                "url": "https://drive.google.com/file/d/1GxfSS7kKZ39hze306wKtM31GA47gLDp0/view?usp=sharing",
                "text": ""
              }
            ]
          }
        ]
      }
    ]
"user_profile": {
      "display_name": "GSDS행정실",
      "name": "snu.gsds",
    }
'''
# https://anaconda.org/pytorch/faiss-cpu

import json
import datetime
import torch
import os

import faiss
import numpy as np


from transformers import BertTokenizer, BertModel

# https://github.com/monologg/KoBERT-Transformers
from transformers import BertModel, DistilBertModel
from tokenization_kobert import KoBertTokenizer
model = BertModel.from_pretrained('monologg/kobert')
distilbert_model = DistilBertModel.from_pretrained('monologg/distilkobert')
tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert') # monologg/distilkobert도 동일
# tokenizer.tokenize("[CLS] 한국어 모델을 공유합니다. [SEP]")
# ['[CLS]', '▁한국', '어', '▁모델', '을', '▁공유', '합니다', '.', '[SEP]']
# tokenizer.convert_tokens_to_ids(['[CLS]', '▁한국', '어', '▁모델', '을', '▁공유', '합니다', '.', '[SEP]'])
# [2, 4958, 6855, 2046, 7088, 1050, 7843, 54, 3]

folder = 'students'
# general, 인턴십등광고홍보게시글, gsds-cluster, random 
# resource, students
folder_path = f'./export/{folder}'
file_path = f'./export/{folder}/attachments'



def parse_date_from_filename(filename):
    # 파일 이름에서 날짜 부분 추출
    # 예: '2023-03-15.json'에서 '2023-03-15'
    date_part = filename.split('.')[0]
    # 날짜 형식에 맞게 파싱
    return datetime.datetime.strptime(date_part, '%Y-%m-%d')

json_files = [pos_json for pos_json in os.listdir(folder_path) if pos_json.endswith('.json')]
json_files.sort(key=lambda x: parse_date_from_filename(x))

def bert_vectorize(text):

    # text to token
    inputs = tokenizer(text, return_tensors = 'pt', max_length = 512, truncation = True, padding = 'max_length')
    # generate vector
    with torch. no_grad():
        outputs = model(**inputs)
    # 평균 풀링을 사용하여 문장 벡터 생성???
    embeddings = outputs.last_hidden_state.mean(1).squeeze().numpy()
    return embeddings

# def create_faiss_index(dimension, data_vectors):
#     """
#     FAISS 인덱스를 생성하고 반환합니다.
#     :param dimension: 벡터의 차원.
#     :param data_vectors: 인덱스에 추가할 벡터들.
#     :return: 생성된 FAISS 인덱스.
#     """
#     index = faiss.IndexFlatL2(dimension)  # L2 거리를 위한 IndexFlatL2 사용
#     if data_vectors is not None:
#         index.add(data_vectors)  # 벡터를 인덱스에 추가
#     return index

def create_faiss_index(dimension):
    """
    주어진 차원에 맞는 FAISS 인덱스를 생성합니다.
    """
    return faiss.IndexFlatL2(dimension)

# FAISS 인덱스 초기화
dimension = 768  # BERT 벡터의 차원, 모델에 따라 다를 수 있음
#faiss_index = create_faiss_index(dimension)


def extract_text_from_blocks(blocks):
    """
    'blocks' 필드에서 텍스트를 추출합니다.
    """
    text = ""
    if not blocks:  # blocks가 None이거나 빈 리스트인 경우
        return text
    for block in blocks:
        if 'elements' in block:
            for element in block['elements']:
                if 'elements' in element:
                    for item in element['elements']:
                        if item['type'] == 'text':
                            text += item['text'] + " "
    return text.strip()



# 각 JSON 파일을 처리한 후 벡터를 얻은 후
vectors = []  # 모든 벡터를 저장할 리스트

dimension = 768
faiss_index = create_faiss_index(dimension)
vector_to_time_mapping = []


for json_file in json_files:
    file_path = os.path.join(folder_path, json_file)
    with open(file_path, 'r') as file:
        data = json.load(file)
        # 데이터 전처리 및 변환
        #print("데이터 전처리 및 변환 시작")
        for item in data:
            #user = item.get('user', '')
            #display_name = item.get('user_profile', {}).get('display_name', '')
            text = item.get('text', '') + " " + extract_text_from_blocks(item.get('blocks', []))
            ts = item.get('ts', '')
            thread_ts = item.get('thread_ts', None)  # thread_ts가 없는 경우 None으로 설정
            
            #print("벡터라이즈 함수 전")
            vector = bert_vectorize(text)
           # print("벡터라이즈 함수 후")
           
           # vectors.append(vector)  # 리스트에 벡터 추가

            # FAISS 인덱스에 벡터 추가 및 시간 데이터 매핑
            faiss_index.add(np.array([vector]).astype('float32'))
            vector_to_time_mapping.append({'ts': ts, 'thread_ts': thread_ts})
# 유사한 벡터를 찾은 후, 관련 시간 데이터 조회
# 예시: 특정 벡터와 가장 유사한 벡터 5개 검색
# search_vector = np.array([some_vector]).astype('float32')
# D, I = faiss_index.search(search_vector, k=5)
# for i in I[0]:
#     print(vector_to_time_mapping[i])  # 각 인덱스에 해당하는 시간 데이터 출력
               
    

#인덱스 저장
faiss.write_index(faiss_index, f"faiss_index_file_{folder}")  # 파일 이름을 원하는 대로 설정하세요

# 시간 데이터 저장
with open(f"vector_to_time_mapping_{folder}.json", "w") as f:  # 파일 이름을 원하는 대로 설정하세요
    json.dump(vector_to_time_mapping, f, ensure_ascii=False, indent=4)

# 이후에 필요할 때 다음과 같이 로드할 수 있습니다.
# faiss_index = faiss.read_index("faiss_index_file")
# with open("vector_to_time_mapping.json", "r") as f:
#     vector_to_time_mapping = json.load(f)




# # JSON 파일 열기 및 파싱
# with open('your_file.json', 'r') as file:
#     data = json.load(file)

# print(f"vector = ", vector)
# print(f"faiss_index = ", faiss_index)
# print(f"vector_to_time_mapping" , vector_to_time_mapping)