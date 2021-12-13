# Loan repayment prediction model 
# 대출 위험도 예측 모델



---

## Contents

* [data](https://github.com/jinokiim/Loan-repayment-prediction-model/tree/main/data)
   * [cs-training.csv](https://github.com/jinokiim/Loan-repayment-prediction-model/blob/main/data/cs-training.csv)
* [saved_model](https://github.com/jinokiim/Loan-repayment-prediction-model/tree/main/saved_model)
   * [gradient_boosting_best.pkl](https://github.com/jinokiim/Loan-repayment-prediction-model/blob/main/saved_model/gradient_boosting_best.pkl)
   * [random_forest_best.pkl](https://github.com/jinokiim/Loan-repayment-prediction-model/blob/main/saved_model/random_forest_best.pkl)
   * [xgb_best.pkl](https://github.com/jinokiim/Loan-repayment-prediction-model/blob/main/saved_model/xgb_best.pkl)
* [01_데이터전처리.ipynb](https://github.com/jinokiim/Loan-repayment-prediction-model/blob/main/01_%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%A0%84%EC%B2%98%EB%A6%AC.ipynb)
* [02_모델링.ipynb](https://github.com/jinokiim/Loan-repayment-prediction-model/blob/main/02_%EB%AA%A8%EB%8D%B8%EB%A7%81.ipynb)


## Process
* 01. 데이터 전처리
* 02. 모델링

---
## 01. 데이터 전처리

### 데이터셋 읽기

```python
# 데이터파일 읽기
data = pd.read_csv('data/cs-training.csv')
data.shape
```
    (150000, 12)

```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>SeriousDlqin2yrs</th>
      <th>RevolvingUtilizationOfUnsecuredLines</th>
      <th>age</th>
      <th>NumberOfTime30-59DaysPastDueNotWorse</th>
      <th>DebtRatio</th>
      <th>MonthlyIncome</th>
      <th>NumberOfOpenCreditLinesAndLoans</th>
      <th>NumberOfTimes90DaysLate</th>
      <th>NumberRealEstateLoansOrLines</th>
      <th>NumberOfTime60-89DaysPastDueNotWorse</th>
      <th>NumberOfDependents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0.766127</td>
      <td>45</td>
      <td>2</td>
      <td>0.802982</td>
      <td>9120.0</td>
      <td>13</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>0.957151</td>
      <td>40</td>
      <td>0</td>
      <td>0.121876</td>
      <td>2600.0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>0.658180</td>
      <td>38</td>
      <td>1</td>
      <td>0.085113</td>
      <td>3042.0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>0.233810</td>
      <td>30</td>
      <td>0</td>
      <td>0.036050</td>
      <td>3300.0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>0.907239</td>
      <td>49</td>
      <td>1</td>
      <td>0.024926</td>
      <td>63588.0</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>

- 제공되는 데이터셋의 컬럼명이 이해하기 어렵거나 사용하기 어렵다면 변경하도록 한다.
    - 컬럼명을 소문자로 변경함.



### 결측치 확인 
```python
data.isna().sum()
```




    seriousdlqin2yrs                            0
    revolvingutilizationofunsecuredlines        0
    age                                         0
    numberoftime30-59dayspastduenotworse        0
    debtratio                                   0
    monthlyincome                           29731
    numberofopencreditlinesandloans             0
    numberoftimes90dayslate                     0
    numberrealestateloansorlines                0
    numberoftime60-89dayspastduenotworse        0
    numberofdependents                       3924
    dtype: int64


### 결측치 처리
---
### 이상치 확인 및 처리

#### IQR 기반 이상치 검출
- IQR : 3분위수-1분위수
- 이상치 기준 (rate는 일반적으로 1.5사용)
    - 극단적으로 작은 값 범위
         - 1분위수 + IQR*rate 보다 작은수
    - 극단적으로 큰 값 범위
        - 3분위 + IQR*rate 보다 큰수

### 각 컬럼별 이상치 처리

#### revolvingutilizationofunsecuredlines
- 전체 운용가능한 돈 대비 현재 운용가능한 돈의 비율 (남은신용한도+통장잔고/ 총신용한도+통장잔고)
- 1초과하는 값들을 1로 변경한다.

#### age
- 대출자 나이
- 최소값이 0, 최대값 109
- 중위수로 변환 

#### numberoftime30-59dayspastduenotworse
- 30 ~ 59 간 연체한 횟수
- 96, 98 이상치 존재 
    - 98의 개수가 264개로 어느정도 양이 되므로 유지


#### debtratio 
- 소득 대비 부채비율(대출상환금+생활비/소득)
- 이상치가 아닌 값들 중 최대값으로 대체한다. 

#### monthlyincome
- 월간 소득
- 이상치를 이상치 아닌 값들의 최대 값으로 대체한다.

## 전처리한 data파일 저장

```python
data.to_csv('data/data-v01.csv', index=False)
```
---
## 02. 모델링


### Feature Scaler생성
### Base-line 모델 정의

## GridSearchCV를 이용한 하이퍼파라미터 튜닝
### XGBoost
### GradientBoosting
### RandomForest
### VotingClassifier
- best model들 사용

## Test Set 으로 검증

### Test set 최종 검증결과
- xgboost : 0.8621545381502079
- grandient boosting : 0.8617491633263116
