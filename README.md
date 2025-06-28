# BDD100K CNN Training

간단한 CNN 모델을 사용하여 BDD100K 데이터셋의 객체 분류를 학습하는 프로젝트입니다.

## 구성
- `train.py`: 학습 및 validation 코드
- `model.py`: CNN 모델 정의
- `dataset.py`: BDD100K Dataset 정의 및 전처리
- `test.py`: 저장된 모델을 이용한 테스트 코드 (예정)
- `requirements.txt`: 의존 패키지 목록

## 실행 방법

```bash
pip install -r requirements.txt
python train.py