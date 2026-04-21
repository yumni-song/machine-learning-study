# 📘 서포트 벡터 머신 (SVM) 이론 및 실습

---

## 5.1 선형 SVM 분류
SVM은 클래스 사이에 가장 폭이 넓은 도로를 찾는 **라지 마진 분류(Large Margin Classification)**를 수행합니다.

* **서포트 벡터(Support Vector)**: 결정 경계(도로)의 가장자리에 위치하여 경계를 지탱하는 샘플들입니다.
* **하드 마진(Hard Margin)**: 모든 샘플이 도로 밖에 위치해야 함. 이상치에 매우 민감합니다.
* **소프트 마진(Soft Margin)**: 도로 폭을 넓게 유지하는 것과 **마진 오류(Margin Violation)** 사이의 균형을 맞춥니다.
    * **C 하이퍼파라미터**: 오차 허용 범위 조절. C가 작을수록 도로가 넓어지며 규제가 강해집니다(과대적합 방지).

---

## 5.2 비선형 SVM 분류
선형적으로 분리되지 않는 데이터는 **다항 특성 추가**나 **커널 트릭**을 사용하여 해결합니다.

* **커널 트릭 (Kernel Trick)**: 실제 특성을 추가하지 않고도 고차원 특성을 추가한 것과 같은 효과를 냅니다.
    * **다항식 커널**: `degree`를 통해 복잡한 데이터셋에 대응.
    * **가우스 RBF 커널**: 샘플 간의 유사도를 측정하여 비선형 경계를 생성. $\gamma$(감마)가 커질수록 결정 경계가 개별 샘플에 민감해집니다.

---

## 5.3 SVM 회귀
분류와 달리 **제한된 마진 오류 안에서 도로 안에 가능한 한 많은 샘플이 들어오도록** 학습합니다.
* **$\epsilon$(앱실론)**: 도로의 폭을 결정. 마진 안의 샘플은 오차에 영향을 주지 않는 $\epsilon$-불감(epsilon-insensitive) 성질을 가집니다.

---

## 5.4 SVM 이론 (수학적 목적 함수)

SVM의 훈련은 제약 조건이 있는 최적화 문제를 푸는 과정입니다.

### 5.4.1 하드 마진 선형 SVM 목적 함수
도로의 폭을 최대화하는 것은 가중치 벡터 $\mathbf{w}$의 노름을 최소화하는 것과 같습니다. 모든 샘플이 마진 바깥에 있어야 한다는 제약 조건이 붙습니다.

$$
\begin{aligned}
&\text{minimize}_{\mathbf{w}, b} \quad \frac{1}{2} \mathbf{w}^T \mathbf{w} \\
&\text{subject to} \quad t^{(i)}(\mathbf{w}^T \mathbf{x}^{(i)} + b) \geq 1 \quad (i = 1, 2, \dots, m)
\end{aligned}
$$


### 5.4.2 소프트 마진 선형 SVM 목적 함수
마진 오류를 허용하기 위해 **슬랙 변수(slack variable)** $\zeta^{(i)} \geq 0$를 도입합니다. $\zeta^{(i)}$는 $i$번째 샘플이 마진을 얼마나 위반할지 결정합니다.

$$
\begin{aligned}
&\text{minimize}_{\mathbf{w}, b, \zeta} \quad \frac{1}{2} \mathbf{w}^T \mathbf{w} + C \sum_{i=1}^{m} \zeta^{(i)} \\
&\text{subject to} \quad t^{(i)}(\mathbf{w}^T \mathbf{x}^{(i)} + b) \geq 1 - \zeta^{(i)} \quad \text{and} \quad \zeta^{(i)} \geq 0
\end{aligned}
$$

* **$C$**: 슬랙 변수의 합에 곱해지는 하이퍼파라미터로, 마진 폭과 마진 오류 사이의 트레이드오프를 결정합니다.


---

## 🚀 SVM 모델 선택 가이드

| 모델 | 시간 복잡도 | 커널 지원 | 특징 |
| :--- | :--- | :---: | :--- |
| **LinearSVC** | $O(m \times n)$ | 아니오 | 빠름, `liblinear` 기반 |
| **SVC** | $O(m^2 \times n)$ ~ $O(m^3 \times n)$ | **예** | 커널 트릭 지원, 대규모 데이터에 느림 |
| **LinearSVR** | $O(m \times n)$ | 아니오 | 선형 회귀 전용 |
| **SVR** | $O(m^2 \times n)$ ~ $O(m^3 \times n)$ | **예** | 비선형 회귀 지원 |

---