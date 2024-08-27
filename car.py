import streamlit as st
st.title("Car Sales Price Prediction")
import numpy as np
import pickle
with open("model.pkl",'rb') as file:
    car_model= pickle.load(file)
def car(resale,price,engine_s,horsepow,wheelbas,width,length,curb_wgt,fuel_cap,mpg,lnsales):
    user_data=np.array([[resale,price,engine_s,horsepow,wheelbas,width,length,curb_wgt,fuel_cap,mpg,lnsales]])
    car_prediction= car_model.predict(user_data)
    return car_prediction
resale= st.slider("resale",min_value=0.5,max_value=100.6)
price= st.slider("price",min_value=.5,max_value=100.5)
engine_s= st.slider("engine_s",min_value=.1,max_value=100.1)
horsepow= st.slider("horsepow",min_value=1,max_value=500)
wheelbas= st.slider("wheelbas",min_value=.1,max_value=100.1)
width= st.slider("width",min_value=.5,max_value=100.5)
length= st.slider("length",min_value=.5,max_value=200.5)
curb_wgt= st.slider("curb_wgt",min_value=.3,max_value=100.5)
fuel_cap= st.slider("fuel_cap",min_value=.6,max_value=100.6)
mpg= st.slider("mpg",min_value=1,max_value=100)
lnsales= st.slider("lnsales",min_value=.5,max_value=100.6)
if st.button("Predict"):
    st.write(f" The predicted value is {resale},{price},{engine_s},{horsepow},{wheelbas},{width},{length},{curb_wgt},{fuel_cap},{lnsales}")
    final_value= car(resale,price,engine_s,horsepow,wheelbas,width,length,curb_wgt,fuel_cap,mpg,lnsales)
    st.write(f" final prediction is {final_value[0]}")
    
       



