
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



import streamlit as st
# import xlsxwriter










# 학습파일 불러오기
df_raw2 = pd.read_excel('data/metro_sim_month.xlsx')
df_raw2.head()

st.subheader('LinearRegression 학습 대상 파일 직접 업로드 하기')
st.caption('업로드 하지 않아도 기본 학습 Data-set 으로 작동합니다 ', unsafe_allow_html=False)

# 학습할 파일을 직접 업로드 하고 싶을때
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  df_raw2 = pd.read_excel(uploaded_file)
  st.write(df_raw2)

# df_raw.columns

df_raw2.rename(columns={
    'ACH50':'ACH50_2',
    'Lighting_power_density_':'Lighting_power_density__2',
    'Chiller_COP':'Chiller_COP_2',
    'Pump_efficiency':'Pump_efficiency_2',
    'Fan_total_efficiency':'Fan_total_efficiency_2',
    'heat_recover_effectiveness':'heat_recover_effectiveness_2',
    'AHU_economiser':'AHU_economiser_2',
    'Occupied_floor_area':'Occupied_floor_area_2',
    'Floor':'Floor_2',
    'Basement':'Basement_2',
    'Ground':'Ground_2',
    },inplace=True)


# 독립변수컬럼 리스트
lm_features2 =['ACH50_2', 'Lighting_power_density__2', 'Chiller_COP_2', 'Pump_efficiency_2',
       'Fan_total_efficiency_2', 'heat_recover_effectiveness_2', 'AHU_economiser_2',
       'Occupied_floor_area_2', 'Floor_2', 'Basement_2', 'Ground_2',]

# 종속변수들을 드랍시키고 독립변수 컬럼만 X_data에 저장
X_data = df_raw2[lm_features2]
# X_data
    
X_data = X_data.astype('float')
# 독립변수들을 드랍시키고 종속변수 컬럼만 Y_data에 저장
Y_data = df_raw2.drop(df_raw2[lm_features2], axis=1, inplace=False)
lm_result_features2 = Y_data.columns.tolist()
# lm_result_features

# 로우 데이터 전체로 회귀모델을 만들고 싶을때
# X_train = X_data.copy() 
# y_train = Y_data.copy()

# 학습데이터에서 일부를 분리하여 테스트세트를 만들어 모델을 평가 하고 싶을때
X_train, X_test, y_train, y_test = train_test_split(
  X_data, Y_data, 
  test_size=0.2, 
  random_state=150)

# 학습시키기 모델이름 lr에 저장
# lr = LinearRegression()
lr2 = LinearRegression()
lr2.fit(X_train, y_train)

# 테스트 세트로 예측해보고 예측결과를 평가하기
y_preds = lr2.predict(X_test)



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




print('절편값:',lr2.intercept_)
print('회귀계수값:',np.round(lr2.coef_, 1))
# print('회귀계수값:',lr.coef_)


# 회귀계수를 테이블로 만들어 보기 1 전치하여 세로로 보기
coeff = pd.DataFrame(np.round(lr2.coef_,2), columns=lm_features2).T
# coeff = coeff.reset_index()
# coeff
coeff.columns = lm_result_features2
# coeff = coeff.reset_index()
# coeff = coeff.rename(columns=lm_result_features)
# coeff
st.subheader('LinearRegression 회귀계수')
st.caption('--------', unsafe_allow_html=False)
coeff

# # 회귀계수를 테이블로 만들어 보기 2 그대로 보기
# coeff2 = pd.DataFrame(np.round(lr.coef_,2), \
#     columns=lm_features, 
#     index=[
#         'Room_Electricity', 
#         'Lighting', 'Fans', 
#         'Pumps', 
#         'Heating', 
#         'Cooling',
#         'DHW', 
#         'Electricity_total'])
# # coeff2


# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters_변경후')

def user_input_features2():
    # ACH50 = st.sidebar.slider('ACH50', X_data.ACH50.min(), X_data.ACH50.max(), X_data.ACH50.mean())
    ACH50_2 = st.sidebar.slider('침기율', 0, 50, 25)
    Lighting_power_density__2 = st.sidebar.slider('Lighting_power_density_', 3, 20, 7)
    Chiller_COP_2 = st.sidebar.slider('Chiller_COP', 4, 9, 6)
    Pump_efficiency_2 = st.sidebar.slider('Pump_efficiency', 0.0, 1.0, 0.7)
    Fan_total_efficiency_2 = st.sidebar.slider('Fan_total_efficiency', 0.0, 1.0, 0.7)
    heat_recover_effectiveness_2 = st.sidebar.slider('heat_recover_effectiveness', 0.0, 1.0, 0.7)
    AHU_economiser_2 = st.sidebar.select_slider('AHU_economiser', options=[0, 1])
    Occupied_floor_area_2 = st.sidebar.slider('Occupied_floor_area', 5000, 10000, 6000)
    Floor_2 = st.sidebar.select_slider('Floor 규모선택', options=[1,2,3])
    Basement_2 = st.sidebar.select_slider('지상유무', options=[0, 1])
    Ground_2 = st.sidebar.select_slider('지하유무', options=[0, 1])

    data = {'ACH50_2': ACH50_2,
            'Lighting_power_density__2': Lighting_power_density__2,
            'Chiller_COP_2': Chiller_COP_2,
            'Pump_efficiency_2': Pump_efficiency_2,
            'Fan_total_efficiency_2': Fan_total_efficiency_2,
            'heat_recover_effectiveness_2': heat_recover_effectiveness_2,
            'AHU_economiser_2': AHU_economiser_2,
            'Occupied_floor_area_2': Occupied_floor_area_2,
            'Floor_2': Floor_2,
            'Basement_2': Basement_2,
            'Ground_2': Ground_2,}
    features2 = pd.DataFrame(data, index=[0])
    return features2

df2 = user_input_features2()

result2 = lr2.predict(df2)
# result
####################################################################################################


##################################################################################################





# result = lr.predict(input)
# result


st.subheader('에너지 사용량 예측값')
st.caption('좌측의 변수항목 슬라이더 조정 ', unsafe_allow_html=False)
st.caption('--------- ', unsafe_allow_html=False)
# 예측값을 데이터 프레임으로 만들어 보기


# df_month = pd.read_excel('data/month.xlsx')

df_result2 = pd.DataFrame(result2, columns=lm_result_features2).T.rename(columns={0:'kW'})
# df_result
df_result2.reset_index(inplace=True)

# 숫자만 추출해서 행 만들기 
# 숫자+'호' 문자열 포함한 행 추출해서 행 만들기 df['floor'] = df['addr'].str.extract(r'(\d+호)')
df_result2['Month'] = df_result2['index'].str.extract(r'(\d+)')
df_result2['index'] = df_result2['index'].str.slice(0,-3)
df_result2
# df_sum = df_result.groupby(['index']).sum()
# df_sum2 = df_result.groupby(['index','Month']).sum()
# df_sum2

# df_result.to_excel('result2.xlsx',engine='xlsxwriter')



# 예측값을 데이터 프레임으로 만들어본것을 그래프로 그려보기

st.subheader('사용처별 에너지 사용량 예측값 그래프')
st.caption('--------- ', unsafe_allow_html=False)

fig = px.box(df_result2, x='index', y='kW', title='제목 ', hover_data=['index'],color='index' )
fig.update_xaxes(rangeslider_visible=True)

st.plotly_chart(fig, use_container_width=True)


# 예측값을 데이터 프레임으로 만들어본것을 그래프로 그려보기

st.subheader('월별 에너지 사용량 예측값 그래프')
st.caption('--------- ', unsafe_allow_html=False)

fig = px.bar(df_result2, x='Month', y='kW', title='제목 ', hover_data=['kW'],color='index' )
fig.update_xaxes(rangeslider_visible=True)

st.plotly_chart(fig, use_container_width=True)



# 예측값을 데이터 프레임으로 만들어본것을 그래프로 그려보기

st.subheader('월별 에너지 사용량 예측값 그래프')
st.caption('--------- ', unsafe_allow_html=False)

fig = px.line(df_result2, x='Month', y='kW', title='제목 ', hover_data=['kW'],color='index' )
fig.update_xaxes(rangeslider_visible=True)

st.plotly_chart(fig, use_container_width=True)

