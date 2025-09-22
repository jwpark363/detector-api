import os, json
import face_recognition
import numpy as np
from typing import List, Dict, Tuple

'''
사용법
face_model = FaceRecognition()
face_model.fit(image_folder_name)
name = face_model.whoami(image_file)
name : 이미지 폴더에서 학습한 결과에 따른 일치하는 사람 이름

규칙
1. 이미지명 : 반드시 코드명으로
2. json 파일 : 반드시 하나의 파일 작성
3. json 파일 구성 : '코더':'이름' 형식으로 작성
예)
{
    "1":"Joe Biden",
    "2":"Barack Obama"
}
'''
class FaceRecognition:
    def __init__(self) -> None:
        self._encodings = []
        self._labels = []
        self._names = None
    '''
    이미지 폴더를 이용 학습
    이미지 폴더의 이미지 파일들 읽어서 인코딩 테이블, 레이블 테이블, 레이블의 이름 테이블 생성
    '''
    def fit(self,train_folder:str):
        print(f'1. load train folder : {train_folder}')
        train_files = os.listdir(train_folder)
        for step, file in enumerate(train_files):
            print(f'2 - {step+1}. train file : {file} ')
            file_name,file_ext = os.path.splitext(file)
            ## Json 파일 일경우 이름 테이블 생성
            if file_ext == '.json':
                print(f'2 - {step+1} *. name table')
                if self._names is None:
                    with open(os.path.join(train_folder,file)) as f:
                        self._names = json.load(f)
                continue
            encoding = self.__face_encoding(os.path.join(train_folder,file))
            self._labels.append(file_name)
            self._encodings.append(encoding)
        print(f'3. train completed!')
        print('encoding : ',len(self._encodings))
        print('labels : ',self._labels)
        print('names : ',self._names)
        
    ## 이미지의 인코딩 구하기
    def __face_encoding(self,image_file):
        image = face_recognition.load_image_file(image_file)
        encoding = face_recognition.face_encodings(image)
        if len(encoding) >= 1:
            encoding = encoding[0]
        return encoding
    ## 학습된 인코딩 테이블
    def encodings(self) -> List[np.ndarray]:
        return self._encodings
    ## 학습된 레이블
    def labels(self) -> List[str]:
        return self._labels
    def names(self) -> Dict[str,str]:
        return self._names
    ## 이미지 파일과 학습된 인코딩과의 거리
    def distance(self,image_file:str) -> np.ndarray:
        return face_recognition.face_distance(self._encodings,self.__face_encoding(image_file))
    ## 이미지 파일과 학습된 인코딩과의 일치 여부
    def compare(self,image_file:str) -> List[np.bool_]:
        encoding = self.__face_encoding(image_file)
        return face_recognition.compare_faces(self._encodings,encoding)
    ## 이미지와 일치하는 코드와 이름, 없으면 None
    def whoami(self,image_file:str)->Tuple[str,str] | None:
        _distance = self.distance(image_file)
        _compare = self.compare(image_file)
        idx = np.argmin(_distance)
        label_idx = -1
        if _compare[idx]:
            label_idx = idx
        if label_idx == -1:
            return None
        return (self._labels[label_idx], self._names[self._labels[label_idx]])