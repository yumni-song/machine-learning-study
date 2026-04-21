# 📘 모델 훈련: 다항 회귀, 규제 및 분류

## 4.3 & 4.4 다항 회귀와 학습 곡선
데이터가 단순한 직선 형태가 아닐 때, 특성의 거듭제곱을 새로운 특성으로 추가하여 선형 모델을 훈련시키는 방법입니다.

* **PolynomialFeatures**: 사이킷런을 통해 차수를 높여 데이터를 변환합니다.
* **학습 곡선 (Learning Curve)**: 훈련 세트 크기에 따른 오차를 그래프로 나타내어 과대/과소적합을 판별합니다.
    * **과대적합(Overfitting)**: 훈련 오차는 낮으나 검증 오차가 높음. 두 곡선 사이의 간격(Gap)이 큼.
    * **과소적합(Underfitting)**: 두 곡선이 모두 높은 오차에서 수렴함.

---

## 4.5 규제가 있는 선형 모델 (Regularization)
모델의 파라미터 가중치를 제한하여 과대적합을 방지하는 방법입니다.

### 4.5.1 릿지 회귀 (Ridge Regression)
비용 함수에 가중치의 제곱($l_2$ 노름)에 비례하는 항을 추가합니다.
* **비용 함수**: $J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + \frac{\alpha}{m} \sum_{i=1}^{n} \theta_i^2$
* **정규 방정식**: $\hat{\boldsymbol{\theta}} = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{A})^{-1} \mathbf{X}^T \mathbf{y}$

### 4.5.2 라쏘 회귀 (Lasso Regression)
비용 함수에 가중치의 절댓값($l_1$ 노름)에 비례하는 항을 추가하여 불필요한 특성을 제거합니다.
* **비용 함수**: $J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + 2\alpha \sum_{i=1}^{n} |\theta_i|$
* **서브그레이디언트 벡터**: $g(\boldsymbol{\theta}, J) = \nabla_{\boldsymbol{\theta}} \text{MSE}(\boldsymbol{\theta}) + \alpha \begin{pmatrix} \text{sign}(\theta_1) \\ \vdots \\ \text{sign}(\theta_n) \end{pmatrix}$

### 4.5.3 엘라스틱넷 (Elastic Net)
릿지와 라쏘 규제를 혼합 비율 $r$로 절충한 모델입니다.
* **비용 함수**: $J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + r\left(2\alpha \sum_{i=1}^{n} |\theta_i|\right) + (1-r)\left(\frac{\alpha}{m} \sum_{i=1}^{n} \theta_i^2\right)$

---

## 4.6 로지스틱 회귀 (Logistic Regression)

### 4.6.1 확률 추정 및 예측
* **확률 추정**: $\hat{p} = h_{\boldsymbol{\theta}}(\mathbf{x}) = \sigma(\boldsymbol{\theta}^T \mathbf{x})$
* **로지스틱 함수 (Sigmoid)**: $\sigma(t) = \frac{1}{1 + \exp(-t)}$
* **예측 모델**: $\hat{y} = \begin{cases} 0 & \hat{p} < 0.5 \text{일 때} \\ 1 & \hat{p} \geq 0.5 \text{일 때} \end{cases}$

### 4.6.2 비용 함수 (로그 손실)
모델이 양성을 양성으로, 음성을 음성으로 잘 예측하도록 파라미터 $\boldsymbol{\theta}$를 찾습니다.
* **단일 샘플 비용**: $c(\boldsymbol{\theta}) = \begin{cases} -\log(\hat{p}) & y=1 \text{일 때} \\ -\log(1-\hat{p}) & y=0 \text{일 때} \end{cases}$
* **전체 비용 함수 (로그 손실)**: $J(\boldsymbol{\theta}) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{p}^{(i)}) + (1-y^{(i)}) \log(1-\hat{p}^{(i)})]$
* **편도함수**: $\frac{\partial}{\partial \theta_j} J(\boldsymbol{\theta}) = \frac{1}{m} \sum_{i=1}^{m} (\sigma(\boldsymbol{\theta}^T \mathbf{x}^{(i)}) - y^{(i)})x_j^{(i)}$

---

## 4.6.4 소프트맥스 회귀 (Softmax Regression)
다중 클래스 분류를 직접 지원하는 로지스틱 회귀의 확장판입니다.

* **클래스 $k$에 대한 점수**: $s_k(\mathbf{x}) = (\boldsymbol{\theta}^{(k)})^T \mathbf{x}$
* **소프트맥스 함수**: $\hat{p}_k = \sigma(\mathbf{s}(\mathbf{x}))_k = \frac{\exp(s_k(\mathbf{x}))}{\sum_{j=1}^{K} \exp(s_j(\mathbf{x}))}$
* **최종 예측**: $\hat{y} = \text{argmax}_k \sigma(\mathbf{s}(\mathbf{x}))_k$
* **크로스 엔트로피 비용 함수**: $J(\mathbf{\Theta}) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log(\hat{p}_k^{(i)})$
* **그레이디언트 벡터**: $\nabla_{\boldsymbol{\theta}^{(k)}} J(\mathbf{\Theta}) = \frac{1}{m} \sum_{i=1}^{m} (\hat{p}_k^{(i)} - y_k^{(i)})\mathbf{x}^{(i)}$

---