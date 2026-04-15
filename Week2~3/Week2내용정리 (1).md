# 📘 머신러닝 프로젝트 처음부터 끝까지

---

## 2.1 실제 데이터로 작업하기
- **사용 데이터셋**: StatLib 저장소의 **캘리포니아 주택 가격(California Housing Prices)** 데이터셋을 사용

---

## 2.2 큰 그림 보기

### 2.2.1 문제 정의
- **목표**: 구역 데이터를 기반으로 **중간 주택 가격**을 예측하는 모델 훈련
- **방법**: 수집된 정보를 취합하여 시스템 설계

### 2.2.2 성능 측정 지표 선택
- **RMSE (Root Mean Square Error)**: 회귀 문제의 전형적인 성능 지표 (평균 제곱근 오차)
- **MAE (Mean Absolute Error)**: 이상치가 많은 경우 고려 (평균 절대 오차)
- **공통점**: 예측값 벡터와 타깃값 벡터 사이의 거리를 측정

$$RMSE(\mathbf{X}, h) = \sqrt{\frac{1}{m}\sum_{i=1}^{m}\left(h(\mathbf{x}^{(i)}) - y^{(i)}\right)^2}$$

### 2.2.3 가정 검사
- 지금까지 세운 가설들을 나열하고 타당성 검토

---

## 2.3 데이터 가져오기

### 📌 데이터 확인 메서드
- `head()`: 데이터의 상위 행 확인
- `info()`: 전체 행 수, 데이터 타입, 결측치(null) 유무 확인
- `value_counts()`: 카테고리별 데이터 개수 확인
- `describe()`: 숫자형 특성의 요약 정보(평균, 표준편차, 백분위수 등) 확인
- `hist()`: 숫자형 특성별 히스토그램 출력

---

### 🧪 테스트 세트 만들기

#### ⚠️ 주의사항
- **데이터 스누핑(Data Snooping) 편향**: 테스트 데이터를 미리 보면 모델 성능이 과대평가될 위험이 있음
- **랜덤 분할의 문제**: 실행 시마다 세트가 변하며, 업데이트 시 이전 데이터와 비교가 어려움

#### 📌 안정적인 분할 및 샘플링
- **재현성 확보**: `random_state`를 설정해 항상 동일한 분할 유지
- **계층적 샘플링 (Stratified Sampling)**: 전체 데이터의 비율(예: 소득 분포)을 유지하며 분할
- **주요 도구**:
  - `train_test_split()`: 데이터 분할 함수
  - `pd.cut()`: 연속형 데이터를 카테고리로 변환
  - `StratifiedShuffleSplit`: 계층별 분할기

#### 💻 주요 코드 예시
```python
# 계층적 샘플링을 이용한 분할
from sklearn.model_selection import train_test_split

strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

# 임시로 만든 income_cat 특성 제거
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
```

---

## 2.4 데이터 이해를 위한 탐색과 시각화
- **원칙**: 테스트 세트는 떼어놓고 **훈련 세트**에 대해서만 탐색 수행
- **복사본 생성**: 원본 데이터를 보존하기 위해 복사본을 만들어 실험

### 2.4.1 지리적 데이터 시각화하기
- **산점도 활용**: 위도(latitude)와 경도(longitude) 시각화
- **밀집도 파악**: `alpha` 옵션을 조절하여 데이터가 밀집된 영역 강조
- **가격 및 인구 표현**: 
  - 원의 크기(`s`): 인구수
  - 색상(`c`): 주택 가격 (`cmap="jet"` 사용)



### 2.4.2 상관관계 조사하기
- **`corr()` 메서드**: 표준 상관계수(Pearson's r) 계산
- **`scatter_matrix`**: 숫자형 특성 간의 산점도 행렬 시각화
- **특징**: 선형적인 상관관계만 측정 가능

### 2.4.3 특성 조합으로 실험하기
- 여러 특성을 조합하여 새로운 특성을 생성 (예: 가구당 방 개수)
- 새로운 조합과 타깃 변수 간의 상관관계 재확인