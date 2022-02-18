import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
mobile_df = pickle.load(open('mobile_df.pkl', 'rb'))

st.title("Mobile Price Predictor")

brand = st.selectbox('Brand', mobile_df['brand'].unique())
ram = st.selectbox('RAM(in GB)', [1, 2, 3, 4, 6, 8, 12])
color = st.selectbox('Color', mobile_df['base_color'].unique())
processor = st.selectbox('Processor', mobile_df['processor'].unique())
rom = st.selectbox('ROM(in GB)', [8, 16, 32, 64, 128, 256, 512])
display_size = st.number_input('Display Size')
screen = st.selectbox('Screen Size', mobile_df['screen_size'].unique())
num_rear_camera = st.selectbox('rear camera', [1, 2, 3, 4])
front_camera = st.selectbox('front camera', [1, 2, 3])
ratings = st.number_input('ratings')
device = st.selectbox('Device', mobile_df["Device_type"].unique())

if st.button('Predict Price'):
    query = np.array(
        [brand, color, processor, screen, rom, ram, display_size, num_rear_camera, front_camera, ratings, device])
    query = query.reshape(1, 11)

    st.title("The predicted price of this configuration is " + str(int(pipe.predict(query)[0])))
