#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 22:07:54 2023

@author: dsimbandumwe
"""

#import numpy as np
import pandas as pd
import glob
#import pandas as pd


path_src = './data/2020'
path = './data'
csv_files = glob.glob(path_src + "/*.csv")
bike_id = 37078



            
            

def process_bike_data():

    # This creates a list of dataframes
    df_list = (pd.read_csv(file) for file in csv_files)
    
    # Concatenate all DataFrames
    big_df   = pd.concat(df_list, ignore_index=True)
    
    
    # filter bike
    bike_df = big_df[big_df['bikeid'] == bike_id]


    # write bike to file
    bike_df.to_csv(path + '/bike_' + str(bike_id) + '.csv')
      
      
      
if __name__ == "__main__":
    process_bike_data()