# 분석전에 필요한 라이브러리들을 불러오기
# 테스트
# plotly라이브러리가 없다면 아래 설치
# conda install -c plotly plotly=4.12.0
# conda install -c conda-forge cufflinks-py
# conda install seaborn
   
import glob 
import os
import sys, subprocess
from subprocess import Popen, PIPE
import numpy as np
import pandas as pd

import streamlit as st
import sklearn
import seaborn as sns
# sns.set(font="D2Coding") 
# sns.set(font="Malgun Gothic") 
# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats("retina")
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go 
import chart_studio.plotly as py
import cufflinks as cf
# # get_ipython().run_line_magic('matplotlib', 'inline')


# # Make Plotly work in your Jupyter Notebook
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# init_notebook_mode(connected=True)
# # Use Plotly locally
cf.go_offline()


# 사이킷런 라이브러리 불러오기 _ 통계, 학습 테스트세트 분리, 선형회귀등
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_log_error


st.set_page_config(layout="wide", page_title="LM_Health_center")

# 학습파일 불러오기
df_raw = pd.read_excel('data/hc.xlsx')


st.subheader('LinearRegression 학습 대상 파일 직접 업로드 하기')
st.caption('업로드 하지 않아도 기본 학습 Data-set 으로 작동합니다 ', unsafe_allow_html=False)

# 학습할 파일을 직접 업로드 하고 싶을때
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  df_raw = pd.read_excel(uploaded_file)
  st.write(df_raw)

# df_raw.columns
df_raw2 = df_raw.copy()


# Alt 용 독립변수 데이터셋 컬럼명 수정
df_raw2 = df_raw2.rename(columns={
    '외벽':'외벽_2',
    '지붕':'지붕_2',
    '바닥':'바닥_2',
    '창호열관류율':'창호열관류율_2',
    'SHGC':'SHGC_2',
    '문열관류율':'문열관류율_2',
    '난방효율':'난방효율_2',
    ' 냉방효율':' 냉방효율_2',
    '급탕효율':'급탕효율_2',
    '조명밀도':'조명밀도_2',
    '중부1':'중부1_2',
    '중부2':'중부2_2',
    '남부':'남부_2',
    '제주':'제주_2',
    })


# 독립변수컬럼 리스트
lm_features =[
    '외벽',
    '지붕',
    '바닥',
    '창호열관류율',
    'SHGC',
    '문열관류율',
    '난방효율',
    ' 냉방효율',
    '급탕효율',
    '조명밀도',
    '중부1',
    '중부2',
    '남부',
    '제주',]

# Alt 용 독립변수 데이터셋 컬럼명 리스트
lm_features2 =[
    '외벽_2',
    '지붕_2',
    '바닥_2',
    '창호열관류율_2',
    'SHGC_2',
    '문열관류율_2',
    '난방효율_2',
    ' 냉방효율_2',
    '급탕효율_2',
    '조명밀도_2',
    '중부1_2',
    '중부2_2',
    '남부_2',
    '제주_2',]

# 종속변수들을 드랍시키고 독립변수 컬럼만 X_data에 저장
X_data = df_raw[lm_features]
X_data2 = df_raw2[lm_features2]


# X_data 들을 실수로 변경
X_data = X_data.astype('float')
X_data2 = X_data2.astype('float')

# 독립변수들을 드랍시키고 종속변수 컬럼만 Y_data에 저장
Y_data = df_raw.drop(df_raw[lm_features], axis=1, inplace=False)
Y_data2 = df_raw2.drop(df_raw2[lm_features2], axis=1, inplace=False)
lm_result_features = Y_data.columns.tolist()
lm_result_features2 = Y_data2.columns.tolist()


# 학습데이터에서 일부를 분리하여 테스트세트를 만들어 모델을 평가 학습8:테스트2
X_train, X_test, y_train, y_test = train_test_split(
  X_data, Y_data , 
  test_size=0.2, 
  random_state=150)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
  X_data2, Y_data2 , 
  test_size=0.2, 
  random_state=150)

# 학습 모듈 인스턴스 생성
lr = LinearRegression() 
lr2 = LinearRegression()

# 인스턴스 모듈에 학습시키기
lr.fit(X_train, y_train)
lr2.fit(X_train2, y_train2)

# 테스트 세트로 예측해보고 예측결과를 평가하기
y_preds = lr.predict(X_test)
y_preds2 = lr2.predict(X_test2)

mse = mean_squared_error(y_test, y_preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_preds)
mape = mean_absolute_percentage_error(y_test, y_preds)

# Mean Squared Logarithmic Error cannot be used when targets contain negative values.
# msle = mean_squared_log_error(y_test, y_preds)
# rmsle = np.sqrt(msle)

print('MSE : {0:.3f}, RMSE : {1:.3f}'.format(mse, rmse))
print('MAE : {0:.3f}, MAPE : {1:.3f}'.format(mae, mape))
# print('MSLE : {0:.3f}, RMSLE : {1:.3f}'.format(msle, rmsle))

print('Variance score(r2_score) : {0:.3f}'.format(r2_score(y_test, y_preds)))
r2 = r2_score(y_test, y_preds)


st.subheader('LinearRegression 모델 성능')
st.caption('--------', unsafe_allow_html=False)

col1, col2 = st.columns(2)
col1.metric(label='Variance score(r2_score)', value = np.round(r2, 3))
col2.metric(label='mean_squared_error', value = np.round(mse, 3))

col3, col4 = st.columns(2)
col3.metric(label='root mean_squared_error', value = np.round(rmse, 3))
col4.metric(label='mean_absolute_error', value = np.round(mae, 3))

st.metric(label='mean_absolute_percentage_error', value = np.round(mape, 3))


# print('절편값:',lr.intercept_)
# print('회귀계수값:',np.round(lr.coef_, 1))


# 회귀계수를 테이블로 만들어 보기 1 전치하여 세로로 보기 (ipynb 확인용)
coeff = pd.DataFrame(np.round(lr.coef_,2), columns=lm_features).T
coeff2 = pd.DataFrame(np.round(lr.coef_,2), columns=lm_features2).T

coeff.columns = lm_result_features
coeff2.columns = lm_result_features2

st.subheader('LinearRegression 회귀계수')
st.caption('--------', unsafe_allow_html=False)
coeff
# coeff2


# Sidebar
# Header of Specify Input Parameters

# base 모델 streamlit 인풋
st.sidebar.header('Specify Input Parameters_BASE')

def user_input_features():
    # ACH50 = st.sidebar.slider('ACH50', X_data.ACH50.min(), X_data.ACH50.max(), X_data.ACH50.mean())
    외벽= st.sidebar.slider('외벽', 0.0, 2.0, 0.466 )
    지붕 = st.sidebar.slider('지붕', 0.0, 2.0, 0.264)
    바닥 = st.sidebar.slider('바닥', 0.0, 2.0, 0.466)
    창호열관류율 = st.sidebar.slider('창호열관류율', 0.0, 4.0, 3.0 )
    SHGC = st.sidebar.slider('SHGC', 0.0, 2.0, 0.688)
    문열관류율 = st.sidebar.slider('문열관류율', 0.0, 4.0, 3.0 )
    난방효율 = st.sidebar.slider('난방효율', 0.0, 100.0, 87.0)
    냉방효율 = st.sidebar.slider(' 냉방효율', 0.0, 4.0, 3.0 )
    급탕효율 = st.sidebar.slider('급탕효율', 0.0, 100.0, 87.0 )
    조명밀도 = st.sidebar.slider('조명밀도',  0.0, 20.0, 10.0, )
    중부1 = st.sidebar.select_slider('중부1', options=[0, 1])
    중부2 = st.sidebar.select_slider('중부2', options=[0, 1])
    남부 = st.sidebar.select_slider('남부', options=[0, 1])
    제주 = st.sidebar.select_slider('제주', options=[0, 1])

    data = {'외벽': 외벽,
            '지붕': 지붕,
            '바닥': 바닥,
            '창호열관류율': 창호열관류율,
            'SHGC': SHGC,
            '문열관류율': 문열관류율,
            '난방효율': 난방효율,
            ' 냉방효율': 냉방효율,
            '급탕효율': 급탕효율,
            '조명밀도': 조명밀도,
            '중부1': 중부1,
            '중부2': 중부2,
            '남부': 남부,
            '제주': 제주,}

    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()
result = lr.predict(df_input)



# ALT 모델 streamlit 인풋
st.sidebar.header('Specify Input Parameters_변경후')

def user_input_features2():
    # ACH50 = st.sidebar.slider('ACH50', X_data.ACH50.min(), X_data.ACH50.max(), X_data.ACH50.mean())
    외벽_2= st.sidebar.slider('외벽_2', 0.0, 2.0, 0.466 )
    지붕_2 = st.sidebar.slider('지붕_2', 0.0, 2.0, 0.264)
    바닥_2 = st.sidebar.slider('바닥_2', 0.0, 2.0, 0.466)
    창호열관류율_2 = st.sidebar.slider('창호열관류율_2', 0.0, 4.0, 3.0 )
    SHGC_2 = st.sidebar.slider('SHGC_2', 0.0, 2.0, 0.688)
    문열관류율_2 = st.sidebar.slider('문열관류율_2', 0.0, 4.0, 3.0 )
    난방효율_2 = st.sidebar.slider('난방효율_2', 0.0, 100.0, 87.0)
    냉방효율_2 = st.sidebar.slider(' 냉방효율_2', 0.0, 4.0, 3.0 )
    급탕효율_2 = st.sidebar.slider('급탕효율_2', 0.0, 100.0, 87.0 )
    조명밀도_2 = st.sidebar.slider('조명밀도_2',  0.0, 20.0, 10.0, )
    중부1_2 = st.sidebar.select_slider('중부1_2', options=[0, 1])
    중부2_2 = st.sidebar.select_slider('중부2_2', options=[0, 1])
    남부_2 = st.sidebar.select_slider('남부_2', options=[0, 1])
    제주_2 = st.sidebar.select_slider('제주_2', options=[0, 1])

    data2 = {'외벽_2': 외벽_2,
            '지붕_2': 지붕_2,
            '바닥_2': 바닥_2,
            '창호열관류율_2': 창호열관류율_2,
            'SHGC_2': SHGC_2,
            '문열관류율_2': 문열관류율_2,
            '난방효율_2': 난방효율_2,
            ' 냉방효율_2': 냉방효율_2,
            '급탕효율_2': 급탕효율_2,
            '조명밀도_2': 조명밀도_2,
            '중부1_2': 중부1_2,
            '중부2_2': 중부2_2,
            '남부_2': 남부_2,
            '제주_2': 제주_2,}
            
    features2 = pd.DataFrame(data2, index=[0])
    return features2

df2_input = user_input_features2()

result2 = lr2.predict(df2_input)


st.subheader('에너지 사용량 예측값')
st.caption('좌측의 변수항목 슬라이더 조정 ', unsafe_allow_html=False)
st.caption('--------- ', unsafe_allow_html=False)

# 예측된 결과를 데이터 프레임으로 만들어 보기
df_result = pd.DataFrame(result, columns=lm_result_features).T.rename(columns={0:'kW'})
df_result2 = pd.DataFrame(result2, columns=lm_result_features2).T.rename(columns={0:'kW'})


df_result['Alt'] = 'BASE'
df_result2['Alt'] = 'Alt_1'
# df_result['kW/m2'] = df_result['kW'] / df_input['Occupied_floor_area'][0]
# df_result2['kW/m2'] = df_result2['kW'] / df2_input['Occupied_floor_area_2'][0]


df_result
df_result2

df_concat = pd.concat([df_result,df_result2])




# 추세에 따라 음수값이 나오는것은 0으로 수정
cond1 = df_concat['kW'] < 0
df_concat.loc[cond1,'kW'] = 0

st.checkbox("Use container width _ BASE", value=False, key="use_container_width")
st.dataframe(df_concat, use_container_width=st.session_state.use_container_width)

df_concat = df_concat.reset_index(drop=False)
df_concat = df_concat.round(2)


# 예측값을 데이터 프레임으로 만들어본것을 그래프로 그려보기

st.subheader('사용처별 에너지 사용량 예측값 그래프')
st.caption('--------- ', unsafe_allow_html=False)


fig = px.bar(df_concat, x='index', y='kW', title='BASE_ALT 원별비교 Bar', hover_data=['kW'],   color='Alt' )
fig.update_xaxes(rangeslider_visible=True)
fig.update_layout(barmode='group') #alt별 구분
# fig
st.plotly_chart(fig, use_container_width=True)


fig = px.bar(df_concat, x='Alt', y='kW', title='BASE_ALT 원별비교 Bar', hover_data=['kW'],   color='index' )
fig.update_xaxes(rangeslider_visible=True)
fig.update_layout(barmode='group') #alt별 구분
# fig
st.plotly_chart(fig, use_container_width=True)

df_groupby_sum = df_concat.groupby('Alt')['kW'].sum()
st.dataframe(df_groupby_sum)


df_concat