## import libraries

import pandas as pd
import numpy as np
import xgboost as xgb

np.random.seed(2018)

## read data

trn = pd.read_csv('./input/train_ver2.csv')
tst = pd.read_csv('./input/test_ver2.csv')

## data preprocessing

# 제품 변수를 별도로 저장해 놓는다.
prods = trn.columns[24:].tolist()

# 제품 변수 결측값을 미리 0으로 대체한다.
trn[prods] = trn[prods].fillna(0.0).astype(np.int8)

# 24개 제품 중 하나도 보유하지 않은 고객 데이터를 제거한다.
no_product = trn[prods].sum(axis=1) == 0
trn = trn[~no_product]

# 훈련 데이터와 테스트 데이터를 통합한다. 테스트 데이터에 없는 제품 변수는 0으로 채운다.
for col in trn.columns[24:]:
    tst[col] = 0

df = pd.concat([trn, tst], axis=0)

# 학습에 사용할 변수를 담는 list이다.
features = []

# 범주형 변수를 .factorize() 함수를 통해 label encoding한다.
# https://blog.naver.com/bosongmoon/221807518210
categorical_cols = ['ind_empleado', 'pais_residencia', 'sexo',
                    'tiprel_1mes', 'indresi', 'indext', 'conyuemp',
                    'canal_entrada', 'indfall', 'tipodom', 'nomprov',
                    'segmento']

# na_sentinel : int, default -1 // Value to mark “not found”. // 결측값을 -99로 변환한다.
for col in categorical_cols:
    df[col], _ = df[col].factorize(na_sentinel=-99)

features += categorical_cols

# 수치형 변수의 특이값과 결측값을 -99로 대체하고, 정수형으로 변환한다.
df['age'].replace('Na', -99, inplace=True)
df['age'] = df['age'].astype(np.int8)

df['antiguedad'].replace('    NA', -99, inplace=True)
df['antiguedad'] = df['antiguedad'].astype(np.int8)

df['renta'].replace('    NA', -99, inplace=True)
df['renta'].fillna(-99, inplace=True)
df['renta'] = df['renta'].astype(float).astype(np.int8)

df['indrel_lmes'].replace('P', 5, inplace=True)
df['indrel_lmes'].fillna(-99, inplace=True)
df['indrel_lmes'] = df['indrel_lmes'].astype(float).astype(np.int8)


# 학습에 사용할 수치형 변수를 features에 추가한다.
features += ['age', 'antiguedad', 'renta', 'ind_nuevo', 'indrel'
             'indrel_lems', 'ind_actividad_cliente']


## feature engineering

# (피처 엔지니어링) 두 날짜 변수에서 연도와 월 정보를 추출한다.
# float(x.split('-')[1]), x는 년-월-일 정보를 포함하고 있는데 월은 2번째 있으므로 [1]을 사용한다.
# float("1.2").__str__() 으로 쓸 경우 string으로 변환해서 사용할 수 있다.
# https://technote.kr/252

df['fecha_alta_month'] = df['fecha_alta'].map(lambda x: 0.0 if x.__class__ is float
                                             else float(x.split('-')[1])).astype(np.int8)
df['fecha_alta_year'] = df['fecha_alta'].map(lambda x : 0.0 if x.__class__ is float
                                             else float(x.split('-')[0])).astype(np.int16)
features += ['fecha_alta_month', 'fecha_alta_year']

df['ult_fec_cli_1t_month'] = df['ult_fec_cli_1t'].map(lambda x : 0.0 if x.__class__ is float
                                             else float(x.split('-')[1])).astype(np.int8)
df['ult_fec_cli_1t_year'] = df['ult_fec_cli_1t'].map(lambda x : 0.0 if x.__class__ is float
                                             else float(x.split('-')[0])).astype(np.int16)
features += ['ult_fec_cli_1t_month', 'ult_fec_cli_1t_year']

# 그 외 변수의 결측값은 모두 -99로 대체한다.

df.fillna(-99, inplace=true)

# (피처 엔지니어링) lag-1 데이터를 생성한다.
#

# 날짜를 숫자로 변환하는 함수이다. 2015-01-28은 1, 2016-06-28은 18로 변환된다.
def date_to_int(str_date):
    Y, M, D = [int(a) for a in str_date.strip().split("-")]
    int_date = (int(Y) - 2015) * 12 + int(M)
    return int_date

# 날짜를 숫자로 변환하여 int_date에 저장한다.
df['int_date'] = df['fecha_dato'].map(date_to_int).astype(np.int8)

# 데이터를 복사하고, int_date 날짜에 1을 더하여 lag를 생성한다. 변수명에 _prev를 추가한다.
df_lag = df.copy()
df_lag.columns = [col + '_prev' if col not in ['ncodpers', 'int_date'] else col for col in df.columns]
df_lag['int_date'] += 1

# 원본 데이터와 lag 데이터를 ncodper와 int_date 기준으로 합친다. lag 데이터의 int_date는 1 밀려있기 대문에,
# 저번달의 제품 정보가 삽입된다.

df_trn = df.merge(df_lag, on=['ncodpers', 'int_date'], how='left')

# 메모리 효율을 위해 불필요한 변수를 메모리에서 제거한다.
del df, df_lag

# 저번 달의 제품 정보가 존재하지 않을 경우를 대비하여 0으로 대체한다.
for prod in prods:
    prev = prod + '_prev'
    df_trn[prev].fillna(0, inplace=True)

df_trn.fillna(-99, inplace=True)

# 모델에 들어갈 feature를 구현하는 부분에선 파생변수(lag data)의 24개 값을 그대로 가져가 쓸 것이기 때문에 EDA처럼 처리가 더 있진 않다.
# lag-1 변수를 추가한다.
features += [feature + '_prev' for feature in features]
features += [prod + '_prev' for prod in prods]

## cross validation

# 학습을 위하여 데이터를 훈련, 테스트용으로 분리한다.
# 학습에는 2016-01-28 ~ 2016-02-28 데이터만 사용하고, 검증에는 2016-05-28 데이터를 사용한다.
# 지금은 baseline를 구현하는 것이기 때문에 4개월 데이터, 즉 적은 데이터만 이용해서 구현해본다.

use_dates = ['2016-01-28', '2016-02-28', '2016-03-28', '2016-04-28', '2016-05-28']

trn = df_trn[df_trn['fecha_dato'].isin(use_dates)]
tst = df_trn[df_trn['fecha_dato'] == '2016-06-28']

del df_trn

# 훈련 데이터에서 신규 구매 건수만 추출한다.
# XY['y'] = Y를 통해 XY 데이터프레임에 'y'라는 신규 구매 여부를 boolean형태로 알려주는 라벨 Y를 추가하였다.
# 이렇게 구성해야 target value를 생성할 수 있다.
X = []
Y = []

for i, prod in enumerate(prods):
    prev = prod + '_prev'
    prX = trn[(trn[prod] ==1) & (trn[prev] ==0)]
    prY = np.zeros(prX.shape[0], dtype=np.int8) + i
    X.append(prX)
    Y.append(prY)
XY = pd.concat(X)
Y = np.hstack(Y)
XY['y']= Y

# 훈련, 검증 데이터로 분리한다.
vld_date = '2016-05-28'
XY_trn = XY[XY['fecha_dato'] != vld_date]
XY_vld = XY[XY['fecha_dato'] == vld_date]


## XGboost model & training

# XGBoost 모델 parameter를 설정한다.

param = {
    'bosster' : 'gbtree',
    'max_depth' : 8,
    'nthread' : 4,
    'num_class' : len(prods),
    'objective' : 'multi:softprob',
    'silent' : 1,
    'eval_metric' : 'mlogloss',
    'eta' : 0.1,
    'min_child_weight' : 10,
    'colsample_bytree' : 0.8,
    'colsample_bylevel' : 0.9,
    'seed' : 2018
}

# 훈련, 검증 데이터를 XGBoost 형태로 변환한다.
X_trn = XY_trn.as_matrix(columns=features)
Y_trn = XY_trn.as_matrix(columns=['y'])
dtrn = xgb.DMatrix(Xtrn, label=Y_trn, features_names=features)

X_vld = XY_vld.as_matrix(columns=features)
Y_vld = XY_vld.as_matrix(columns=['y'])
dvld = xgb.DMatrix(X_vld, label=Y_vld, features_names=features)

# XGBoost 모델을 훈련 데이터로 학습한다.
watch_list = [(dtrn, 'train'), (dvld, 'eval')]
model = xgb.train(param, dtrn, num_boost_round=1000, evals=watch_lsit, early_stopping_rouonds=20)

# 학습한 모델을 저장한다.
import pickle

pickle.dump(model, open("model/xgb.baseline.pkl", 'wb'))
best_ntree_limit = model.best_ntree_limit

## evluation