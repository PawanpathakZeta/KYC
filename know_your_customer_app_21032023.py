import streamlit as st
import pandas as pd
import numpy as np    
# import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import os
import sys
from PIL import Image
# from matplotlib import pyplot as plt
import time
import datetime
from io import BytesIO
import uuid
from pyforest import *
import recordlinkage as rl
from recordlinkage.preprocessing import clean
from recordlinkage.index import SortedNeighbourhood
# import recordlinkage as rl
# from recordlinkage.preprocessing import clean
from tqdm import tqdm as tdm
import spacy
from kneed import KneeLocator, DataGenerator as dg
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']

from Back_end.back_end_functions import precleaning, clean_zip, cleaning_cols, Sorted_Neighbourhood_Prediction, elbow_function, merge_dataframes


# 1- Main Window -- Layout Settings------------------------------------------------------------
st.set_page_config(layout="wide")
base="dark"
primaryColor="#BF2A7C" #PINK
backgroundColor="#FFFFFF" #MAIN WINDOW BACKGROUND COLOR (white)
secondaryBackgroundColor="#EBF3FC" #SIDEBAR COLOR (light blue)
textColor="#31333F"
secondaryColor="#F0F2F6" #dark_blue
tertiaryColor ="#0810A6"
light_pink = "#CDC9FA"
plot_blue_colour="#0810A6" #vibrant blue for plots

#List of possible tresholds used in the algorithm
chosen_tresholds = ('0.85','0.95','0.97','0.99')


#
# -----------------------------------------------------------------------------------------------
# 2- Sidebar -- Parameter Settings---------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

# Input Data 
st.sidebar.title("Input Seed File")


## Seed file: this are unknown customers 
Seed_file = st.sidebar.file_uploader('Seed Dataset', type='csv', help='Dataset without email address')

Seed_file_valid_flag = False
if Seed_file is not None:
    # Check MIME type of the uploaded file
    if  Seed_file.name == "Seed_data.csv":
        Seed_df = pd.read_csv(Seed_file)
        Seed_file_valid_flag = True
        st.sidebar.success('Success')
    else:
        st.sidebar.error('Error: please, re-upload file called Seed_data.csv')



# ## Check Data Formats button
check_data = st.sidebar.button("""Check Data""", help = 'Show statistics of your data')


# Persist  unm/cust data statistics
if 'valid_flag' not in st.session_state:
    st.session_state.valid_flag = False

if 'Seed_obj_cols' not in st.session_state:
    st.session_state.Seed_obj_cols = ['']
if 'Seed_num_cols' not in st.session_state:
    st.session_state.Seed_num_cols = ['']

if 'Seed_df_com_cols' not in st.session_state:
    st.session_state.Seed_df_com_cols = pd.DataFrame()


if check_data:
    if (Seed_file_valid_flag == True) :
        Seed_df.columns = map(str.upper, Seed_df.columns)
        commun_cols_u_c = list(Seed_df.columns) 
        if commun_cols_u_c:
            seed_df = Seed_df[commun_cols_u_c]
            st.session_state.valid_flag = True
            Seed_obj_cols = list(Seed_df.select_dtypes(include="object").columns)
            Seed_num_cols = list(Seed_df.select_dtypes(exclude="object").columns)
            Seed_num_cols.insert(0,"")
            st.session_state.seed_df_com_cols = seed_df
            st.session_state.Seed_obj_cols = Seed_obj_cols
            st.session_state.Seed_num_cols = Seed_num_cols
            
            
    else:
        pass
else:
    pass



# Global Parameter Selection 
st.sidebar.title("Global Parameter Selection")
# st.sidebar.subheader('Main field to be matched')

# sidebar columns 5 and 6
col5_sidebar, col6_sidebar= st.sidebar.columns([2, 2])

## Seed  columns

if Seed_file_valid_flag == True:
    selectbox_mss = list([" "])

    # Mail field to be matched
    select_box_Seed_load_main = st.sidebar.selectbox(
        'Source Data',
        options=('FP Only','DC Only','FP + DC'), 
        help = 'Select an option to choose First Party data')
    selectbox_min_clusters_11 = st.sidebar.selectbox('Minimum Number of Clusters',
        options=('2','3','4'), help='Select main field threshold to perfom match',key = 11)
    selectbox_max_clusters_12 = st.sidebar.selectbox('Maximum Number of Clusters',
        options=('5','6','7'), help='Select main field threshold to perfom match', key = 12)
    st.sidebar.markdown(f'<h1 style="color:{tertiaryColor};font-size:16px;">{f"The operation will take less than a minute"}</h1>', unsafe_allow_html=True)

else:
    selectbox_mss = list([" "])
    # Mail field to be matched
    select_box_Seed_main_empty = st.sidebar.selectbox(
        'Source Data',
        options=selectbox_mss, help = 'Please, upload data to start')
    selectbox_min_clusters_11 = st.sidebar.selectbox('Minimum Number of Clusters',
        options=selectbox_mss, help='Select main field threshold to perfom match')
    selectbox_max_clusters_12 = st.sidebar.selectbox('Maximum Number of Clusters',
        options=selectbox_mss, help='Select main field threshold to perfom match')


## Apply Match button
st.sidebar.title("Profile Customer")
match_button = st.sidebar.button("""Apply Profiling""", help='Apply match to your data and show the results')

st.sidebar.title("Download Result Excel")
result_download = st.sidebar.button("""Download""", help='Apply match to your data and show the results')




# -----------------------------------------------------------------------------------------------
# 1- Main Window -- Parameter Settings-----------------------------------------------------------
# ----------------------------------------------------------------------------------------------- 
# st.title(f'My first app {st.__version__}')


# Creating columns 1 and 2
col1, col2 = st.columns([13, 2])

## Zeta Logo
#zeta_logo = Image.open('ZETA_BIG-99e027c9.webp') #white logo 
zeta_logo = Image.open('ZETA_BIG-99e027c92.png') #blue logo 
col2.image(zeta_logo)

## Header
col1.title("Know Your Customer")
"""This app demonstrates understanding Customers"""

# Creating columns 3 and 4
col_space1, col_space2  =  st.columns([2,2])

# Creating columns 3 and 4
col3, col4= st.columns([2, 2])
colna1, colna2 = st.columns([2,2]) # NA message if no NAs

# Create col expand data
col_left_expander, col_right_expander = st.columns(2)

# Creating columns 5 and 6

col_threshold_left, col_threshold_rigt = st.columns([2,2])

# Creating columns 5 and 6

col5, col6 = st.columns([2,2])


## Summary pie charts
colp1, colp2, colp3 = st.columns([2, 2, 2])


# Set num of column algorithm will compare, by default is 1
col_compare_alg = 1




# Display features if data is valid and match button is clicked
if match_button and Seed_file_valid_flag:
    if (selectbox_min_clusters_11 > selectbox_max_clusters_12 ) and ((selectbox_min_clusters_11 != '') or select_box_Seed_load_12 != ''):
        st.error('Please, select two different extra columns to perform match')
    else:
        # Bar progress
        latest_iteration = st.empty()
        bar = st.progress(0)
        latest_iteration.text('Profiling in progress...')
        time.sleep(2)
        # bar = st.progress(0)
        latest_iteration.text('Profiling in Done...')



if st.session_state['valid_flag']:
    #If both files are uploaded print stats of each one
    if  Seed_file_valid_flag==True:
        col_space1.success('Format Check Completed') 
        # print(len(seed_df))   
        df_info = {'Number of rows':[len(st.session_state.seed_df_com_cols)],
                    'Number of columns': [len(st.session_state.seed_df_com_cols.columns)]}
        df_info = pd.DataFrame(df_info).transpose().rename(columns={0:'Seed Data'})
        # print(df_info)
        col_space1.dataframe(df_info)
        
        # Unmatch df
        col_left_expander.write('Seed data')
        with col_left_expander.expander("Expand data and statistics"):
            #Display unmatch df
            st.dataframe(st.session_state.seed_df_com_cols.head(100))
            #Display unmatch df NAs 
            st.write(f'Number of Null values in Seed Dataset: ')
            na_unmt_df = pd.DataFrame(st.session_state.seed_df_com_cols.isna().sum())
            na_unmt_df = na_unmt_df.rename(columns={0:'#NAs'})
            st.dataframe(na_unmt_df)
        
        if len(na_unmt_df[na_unmt_df['#NAs']>0]) == 0:
            colna1.info('No Nulls found in Seed Dataset')


    # Print Seed_file stats if uploaded
    elif Seed_file_valid_flag:
        col_space1.success('Format Check Completed')
        col_left_expander.write('Seed data')
        with col_left_expander.expander("Expand data and statistics"):
            #Display unmatch df
            st.dataframe(st.session_state['Seed_df_com_cols'].head(100))
            #Display unmatch df NAs 
            st.write(f'Number of Nan values in Seed Dataset: ')
            na_unmt_df = pd.DataFrame(st.session_state['Seed_df_com_cols'].isna().sum())
            na_unmt_df = na_unmt_df.rename(columns={0:'#NAs'})
            st.dataframe(na_unmt_df)


    else: 
         pass
else:
    col3.write('')
