import  streamlit as st
import pandas as pd
import numpy as np
import pickle
st.title('Laptop Price Prediction')
st.write('This is a simple web app to predict the price of a laptop based on its specifications')
st.write('Please adjust the values of the features on the left and the predicted price will be shown on the right')
Company=['Lenovo', 'Asus', 'Dell', 'HP', 'MSI', 'Other', 'Acer', 'Toshiba',
       'Apple', 'Samsung']
laptop_type=['Notebook', 'Ultrabook', '2 in 1 Convertible', 'Gaming',
       'Workstation', 'Netbook']
ram=[ 8,  4, 32, 16, 12,  6, 64,  2, 24].sort()
TouchScreen=[0, 1]
ips=[0, 1]
Cpu_brand=['Intel Core i5', 'Intel Core i7', 'Intel Core i3',
       'Other Intel Processor', 'AMD Processor']
HDD=[   0,  500, 1000, 2000,   32,  128]
SSD=[ 256,  512,    0,  128,   32, 1024,  240, 1000,  180,    8,   16,
        768,   64]
Gpu_brand=['Intel', 'Nvidia', 'AMD']
Os=['Windows', 'Mac', 'Other']
model=pickle.load(open('model.pkl','rb'))
col1, col2 = st.columns(2)
with col1:
    company=st.selectbox('Company',Company)
with col2:
    laptop_type=st.selectbox('Laptop Type',laptop_type)
col3, col4 = st.columns(2)
with col3:
    ram=st.slider('Ram',min_value=0,max_value=64,step=4)
with col4:
    weight = st.slider('Weight', min_value=0.0, max_value=3.0, step=0.1)

col5, col6 = st.columns(2)
with col5:
    TouchScreen = st.radio('TouchScreen', TouchScreen)
with col6:
    ips=st.radio('IPS',ips)
col7, col8 = st.columns(2)
with col7:
    Ppi=st.slider('Ppi',min_value=0,max_value=400,step=10)
with col8:
    Cpu_brand=st.selectbox('Cpu Brand',Cpu_brand)
col9, col10 = st.columns(2)
with col9:
    HDD=st.slider('HDD',min_value=0,max_value=2000,step=100)
with col10:
    SSD=st.slider('SSD',min_value=0,max_value=1024,step=32)
col11, col12 = st.columns(2)
with col11:
    Gpu_brand=st.selectbox('Gpu Brand',Gpu_brand)
with col12:
    Os=st.selectbox('Os',Os)

if st.button('Predict'):
    input_data={'Company':company,'TypeName':laptop_type,'Ram':ram,'Weight':weight,'TouchScreen':TouchScreen,'Ips':ips,'Ppi':Ppi,'Cpu_brand':Cpu_brand,'HDD':HDD,'SSD':SSD,'Gpu_brand':Gpu_brand,'Os':Os}
    input_data=pd.DataFrame(input_data,index=[0])
    prediction=model.predict(input_data)
    st.header('The predicted price of the laptop is: '+str(round((prediction[0]*1000)))+' $')

