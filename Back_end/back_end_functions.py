#IMPORT
import recordlinkage as rl
from recordlinkage.preprocessing import clean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kneed import KneeLocator, DataGenerator as dg



#Pre-cleaning function
def precleaning(df):
    # drop columns with 100% nan values
    df = df.dropna(axis=1, how='all')
    
    # drop duplicates
    df.drop_duplicates(inplace=True) 
    
    # make col names capital
    df.columns = df.columns.str.upper()
    
    # create index for the algorithm
    df.insert(0,"ID",range(1,df.shape[0]+1) ,True)
    
    return df



#Clean ZIP code column
def clean_zip(df, col='ZIP'):
    # Clean zip codes that are float type
    if df[col].dtype == float:
        df[col] = df[col].apply(lambda x: str(int(x)) if not pd.isna(x) else x)
        # If there were trailing ".0" after conversion to int, remove them
        df[col] = df[col].str.replace('.0', '')
    
    # Clean zip codes that are string type
    if df[col].dtype == object:
        df[col] = df[col].str.strip()  # Remove any leading/trailing spaces
        df[col] = df[col].str.replace('\D', '')  # Remove any non-digit characters
        df[col] = df[col].apply(lambda x: x.replace('-', '') if x else x)  # Remove hyphens
        df[col] = df[col].apply(lambda x: str(int(x)).zfill(5) if x else x)  # Convert to integer and pad with leading zeros

    return df

        
        

#Cleaning function to clean strings type cols using Record Linkage
def cleaning_cols(df):
    for  col in df.select_dtypes("object").columns:
        df[col]=clean(df[col],
                      lowercase=True, 
                      replace_by_none='[^ \\-\\_A-Za-z0-9]+', 
                      replace_by_whitespace='[\\-\\_]', 
                      strip_accents='unicode', 
                      remove_brackets=True, 
                      encoding='utf-8', 
                      decode_error='ignore')
    return df





def Sorted_Neighbourhood_Prediction(df1, df2, pred_comp=1, threshold=None, method_str=None, method_num=None, scale=None, offset=None, main_field_compare=None, select_box_unmatched_load_11=None, select_box_unmatched_load_12=None):
    #cleaning object cols for model redeability
    df1=cleaning_cols(df1)
    df2=cleaning_cols(df2)
    threshold =float(threshold)
        
    #resetiing index to core customerids of respective datasets
    df1=df1.set_index('ID')
    df2=df2.set_index('ID')
    
    ## creating mathced indexes using SoretdNeighbourHood Approach
    clx = rl.index.SortedNeighbourhood(main_field_compare, window=3)
    clx = clx.index(df1, df2)

    cr = rl.Compare()
    cr.string(main_field_compare, main_field_compare, method=method_str, threshold=threshold, label=main_field_compare)
        
    if select_box_unmatched_load_11:
        cr.numeric(select_box_unmatched_load_11, select_box_unmatched_load_11, scale=scale, offset=offset, label=select_box_unmatched_load_11)
    if select_box_unmatched_load_12:
        cr.numeric(select_box_unmatched_load_12, select_box_unmatched_load_12, method=method_num, scale=scale, offset=offset, label=select_box_unmatched_load_12)

    feature_vectors = cr.compute(clx, df1, df2)
    
    # predictions =feature_vectors[feature_vectors.sum(axis=1) > round(threshold*pred_comp,1)] 
    predictions = feature_vectors[feature_vectors.sum(axis=1) > (threshold*pred_comp)] 

    return feature_vectors, predictions





def merge_dataframes(predictions, unmatched_df, customer_df):
    data_indexes=predictions.reset_index()
    data_indexes=data_indexes["ID_1	ID_2".split("\t")]
    df_v1=data_indexes.merge(unmatched_df,left_on="ID_1",right_on="ID")
    df_v2=data_indexes.merge(customer_df,left_on="ID_2",right_on="ID")
    df_final = df_v1.merge(df_v2,on=["ID_1","ID_2"],how="left",suffixes=('_MTCH', '_UNMTCH'))
    # drop duplicate id
    df_final.drop(["ID_1", "ID_2"],axis=1,inplace=True)
    #drop duplicates
    df_final.drop_duplicates(inplace=True)
    return df_final






def elbow_function(df,
                   x_threshold, 
                   y_num_match, 
                   backgroundColor='color', 
                   plot_blue_colour='plot_blue_colour', 
                   primaryColor='primaryColor', 
                   textColor='textColor'):
    
    #Data handling for elbow chart
    df = pd.DataFrame(df, index = [x_threshold, y_num_match]).transpose() 
    df[x_threshold] =  df[x_threshold].astype(float)
    df[y_num_match] = df[y_num_match].astype(float)
    
    #Elbow chart creation
    elbow_chart, ax = plt.subplots()
    elbow_chart.patch.set_facecolor(backgroundColor)
    ax.set_facecolor(backgroundColor)
    ax.plot(df[x_threshold], df[y_num_match], plot_blue_colour, linewidth = 2.5, marker= 'o')
    plt.fill_between(df[x_threshold], df[y_num_match], color=primaryColor)
    
    # Set the tick values for the x-axis if you want to display only the thresholds on the x axis
#     ax.set_xticks(df[x_threshold])
    
    # Giving x label using xlabel() method with bold setting
    plt.xlabel("Threshold", color=textColor,fontsize =10)
    plt.tick_params(colors=textColor, which='both') 
    # Y label settings, use ontweight='bold' to make font bold
    plt.ylabel("# Matches", color=textColor, rotation='vertical', loc ='center', fontsize =10)
    plt.ylim(((min(df[y_num_match])-50),(max(df[y_num_match])+50)))
    # Giving title to the plotTotal customers 100% match
    plt.title("Number Matched Records per Threshold", color=textColor, fontsize =12)
    
    # Add cosmetics
    ax.spines['bottom'].set_color('lightgrey')
    ax.spines['top'].set_color('lightgrey') 
    ax.spines['right'].set_color('lightgrey')
    ax.spines['left'].set_color('lightgrey')
    plt.show()

    # Calculate Optimal threshold
    # Convert data to numpy arrays
    x = np.array(df[x_threshold])
    y = np.array(df[y_num_match])
    # Find the knee point using the KneeLocator
    s = 1.0 #sensitivity parameter
    best_threshold = None
    while best_threshold is None and s >= 0.1: #if the S is not sensible enough is value is reduced until it finds the best threshold
        kneedle = KneeLocator(x, y, S=s, curve='convex', direction='decreasing')
        best_threshold = kneedle.knee
        s -= 0.1

    return best_threshold, elbow_chart


