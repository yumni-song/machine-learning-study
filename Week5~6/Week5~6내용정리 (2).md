# 📘 모델 훈련: 규제와 로지스틱 회귀

---

## 4.3 & 4.4 다항 회귀와 학습 곡선
* **다항 회귀**: 특성의 거듭제곱을 새로운 특성으로 추가하여 비선형 데이터를 학습.
* **편향/분산 트레이드오프**:
    * **편향(Bias)**: 잘못된 가심으로 인한 오차. 과소적합 유발.
    * **분산(Variance)**: 훈련 데이터의 변동에 과도하게 민감. 과대적합 유발.
    * **줄일 수 없는 오차**: 데이터 자체의 노이즈.

---

## 4.5 규제가 있는 선형 모델

### 4.5.1 릿지 회귀 (Ridge Regression)
비용 함수에 가중치의 제곱($l_2$ 노름) 항을 추가하여 가중치를 작게 유지합니다.

* **릿지 회귀의 비용 함수**:
$$J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + \frac{\alpha}{m} \sum_{i=1}^{n} \theta_i^2$$

* **정규 방정식의 해**:
$$\hat{\boldsymbol{\theta}} = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{A})^{-1} \mathbf{X}^T \mathbf{y}$$

### 4.5.2 라쏘 회귀 (Lasso Regression)
가중치의 절댓값($l_1$ 노름) 항을 추가합니다. 중요한 특성만 남기고 나머지는 0으로 만드는 **특성 선택** 기능이 있습니다.

* **라쏘 회귀의 비용 함수**:
$$J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + 2\alpha \sum_{i=1}^{n} |\theta_i|$$

* **서브그레이디언트 벡터**:
$$g(\boldsymbol{\theta}, J) = \nabla_{\boldsymbol{\theta}} \text{MSE}(\boldsymbol{\theta}) + \alpha \begin{pmatrix} \text{sign}(\theta_1) \\ \text{sign}(\theta_2) \\ \vdots \\ \text{sign}(\theta_n) \end{pmatrix} \quad \text{where } \text{sign}(\theta_i) = \begin{cases} -1 & \theta_i < 0 \\ 0 & \theta_i = 0 \\ +1 & \theta_i > 0 \end{cases}$$

### 4.5.3 엘라스틱넷 (Elastic Net)
릿지와 라쏘를 절충한 모델로, 혼합 비율 $r$을 사용합니다.

* **엘라스틱넷 비용 함수**:
$$J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + r\left(2\alpha \sum_{i=1}^{n} |\theta_i|\right) + (1-r)\left(\frac{\alpha}{m} \sum_{i=1}^{n} \theta_i^2\right)$$

---

## 4.6 로지스틱 회귀 (Logistic Regression)

### 4.6.1 확률 추정 및 예측
* **확률 추정**: $\hat{p} = h_{\boldsymbol{\theta}}(\mathbf{x}) = \sigma(\boldsymbol{\theta}^T \mathbf{x})$
* **로지스틱 함수 (시그모이드)**: $\sigma(t) = \frac{1}{1 + \exp(-t)}$
* **모델 예측**: 
$$\hat{y} = \begin{cases} 0 & \hat{p} < 0.5 \\ 1 & \hat{p} \geq 0.5 \end{cases}$$

### 4.6.2 비용 함수 (로그 손실)
* **단일 샘플 비용**:
$$c(\boldsymbol{\theta}) = \begin{cases} -\log(\hat{p}) & y=1 \text{ 일 때} \\ -\log(1-\hat{p}) & y=0 \text{ 일 때} \end{cases}$$

* **전체 비용 함수 (로그 손실)**:
$$J(\boldsymbol{\theta}) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{p}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)})]$$

* **비용 함수의 편도함수**:
$$\frac{\partial}{\partial \theta_j} J(\boldsymbol{\theta}) = \frac{1}{m} \sum_{i=1}^{m} (\sigma(\boldsymbol{\theta}^T \mathbf{x}^{(i)}) - y^{(i)}) x_j^{(i)}$$

---

## 4.6.4 소프트맥스 회귀 (Softmax Regression)

다중 클래스 분류를 위한 로지스틱 회귀의 확장형입니다.

* **클래스 $k$에 대한 점수**: $s_k(\mathbf{x}) = (\boldsymbol{\theta}^{(k)})^T \mathbf{x}$
* **소프트맥스 함수**: 
$$\hat{p}_k = \sigma(\mathbf{s}(\mathbf{x}))_k = \frac{\exp(s_k(\mathbf{x}))}{\sum_{j=1}^{K} \exp(s_j(\mathbf{x}))}$$

* **예측**: $\hat{y} = \text{argmax}_k \sigma(\mathbf{s}(\mathbf{x}))_k = \text{argmax}_k s_k(\mathbf{x})$
* **크로스 엔트로피 비용 함수**:
$$J(\boldsymbol{\Theta}) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log(\hat{p}_k^{(i)})$$

* **클래스 $k$에 대한 그레이디언트 벡터**:
$$\nabla_{\boldsymbol{\theta}^{(k)}} J(\boldsymbol{\Theta}) = \frac{1}{m} \sum_{i=1}^{m} (\hat{p}_k^{(i)} - y_k^{(i)}) \mathbf{x}^{(i)}$$

---