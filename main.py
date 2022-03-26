"""
A Jarman 2022
"""
from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import streamlit as st
from pycaret.time_series import TSForecastingExperiment

from datasets import DATASET_MAP

st.set_page_config(page_title='AutoTS', page_icon=':stopwatch:')

ts_exp = TSForecastingExperiment()
dataset: Union[pd.DataFrame, pd.Series] = pd.DataFrame()

st.title("Time Series AutoML")
st.text("Developed with PyCaret and Streamlit")
st.header("Data loading")
with st.container() as dataload_container:
    sample_data: bool = st.checkbox("Use sample dataset?")

    if sample_data:
        loader_choice: Callable = st.selectbox(
            "Dataset:", options=DATASET_MAP.keys())
        loader = DATASET_MAP.get(loader_choice, None)
        if loader:
            dataset:pd.Series = loader()
    else:
        file_obj = st.file_uploader(label="Upload a dataset", accept_multiple_files=False,
                                    type=[".csv"])
        if file_obj:
            with st.spinner("Loading data:"):
                dataset:pd.DataFrame = pd.read_csv(file_obj)


    dt_col, target_col = "", ""     
    if not dataset.empty and isinstance(dataset, pd.DataFrame):

        with st.expander("View data:") as viewdata:
            st.dataframe(dataset)

        dataloadcol1, dataloadcol2 = st.columns(2)

        with dataloadcol1:
            dt_col: str = st.selectbox(
                "Select temporal (Date/Datetime) column:",
                options=dataset.columns.tolist())

            if st.button("Process column"):
                dataset[dt_col] = pd.to_datetime(dataset[dt_col])
                dataset = dataset.set_index(dt_col)

        with dataloadcol2:
            target_col: str = st.selectbox(
                "Select target column:",
                options=[i for i in dataset.columns.tolist() if i != dt_col])

    if isinstance(dataset, pd.Series):
        with st.expander("View data:") as viewdata_s:
            view_dataset = dataset.copy()
            view_dataset = view_dataset.to_timestamp()
            st.dataframe(view_dataset.to_frame())


if (not dataset.empty and dt_col and target_col) or isinstance(dataset, pd.Series):
    with st.container() as experiment_container:
        st.header("Experiment Setup")
        forecast_horizon: int = st.slider(label="Forecast Horizon:",
                                          min_value=1, max_value=len(dataset), value=int(len(dataset)/10))
        number_of_folds: int = st.slider(label="Cross validation folds:",
                                         min_value=3, max_value=20, value=5)

    if st.button("Setup experiment:"):

        if isinstance(dataset, pd.DataFrame):
            col_kwargs = {"target": target_col, "index": "dt_col"}
        else:
            col_kwargs = {}
        with st.spinner("Setting up experiment. . .") as experimenter:
            ts_exp.setup(dataset, fh=forecast_horizon,
                         fold=number_of_folds, verbose=False, **col_kwargs)
            # missing_values: bool = False
            # st.table(ts_exp._get_setup_display(
            #     missing_flag=missing_values, imputation_type=None))
            st.success("Experiment setup complete.")

    if ts_exp._setup_ran:
        if st.button("Compare forecast models"):
            with st.spinner("Comparing models") as comparer:
                models = ts_exp.compare_models()
                st.success("Initial model comparison complete")
                st.table(models)
