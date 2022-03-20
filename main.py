from typing import Tuple, List, Optional

import pandas as pd
import streamlit as st
# from pycaret.time_series import TSForecastingExperiment

st.title("Time Series AutoML")
st.text("Developed with PyCaret and Streamlit")
st.header("Data loading")
custom_data:bool = st.checkbox("Use custom dataset?")

df =  pd.DataFrame()
if custom_data:
    file_obj = st.file_uploader(label="Upload a dataset", accept_multiple_files=False,
    type=[".csv"])
    if file_obj:
        with st.spinner("Loading data:"):
            df = pd.read_csv(file_obj)
# else:
#     sample_data:str = st.selectbox("select a sample dataset:", options=["airquality"])
#     df = get_data(sample_data)

if not df.empty:
    with st.expander("View data:") as viewdata:
        st.dataframe(df)

    dataloadcol1, dataloadcol2 = st.columns(2)

    with dataloadcol1 as col:
        dt_col:str = st.selectbox(
            "Select temporal (Date/Datetime) column:", 
        options=df.columns.tolist())
        dt_col_type:str  = st.radio("Type:",options=["Date","Time"])
        
        if st.button("Convert column"):
            df[dt_col] = pd.to_datetime(df[dt_col])
            if dt_col_type == "Date":
                df[dt_col] = df[dt_col].dt.date


    with dataloadcol2 as col:
        target_col:str = st.selectbox(
            "Select target column:", 
        options=[i for i in df.columns.tolist() if i != dt_col])

if not df.empty:
    st.header("Experiment Setup")
    forecast_horizon:int = st.slider(label="Forecast Horizon:",
    min_value=1, max_value=len(df), value=int(len(df)/10))
    number_of_folds:int = st.slider(label="Cross validation folds:",
    min_value=3, max_value=20, value=5)
    with st.expander("Setup experiment:"):
        st.text("Setting up experiment")
    # exp = TSForecastingExperiment()
    # exp.setup(df, fh = forecast_horizon, fold = number_of_folds, target=target_col,)
