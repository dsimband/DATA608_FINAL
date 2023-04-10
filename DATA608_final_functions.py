#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 22:07:54 2023

@author: dsimbandumwe
"""

#import numpy as np
import pandas as pd
#import glob
import requests 
import zipfile
#import pandas as pd


path_src = './data/2020'
path = './data'
#csv_files = glob.glob(path_src + "/*.csv")
#bike_id = 37078
station_lst = ['7617.07']

bike_file = './data/bike.csv'


config_df = pd.DataFrame({
                        'file_url' : ['https://s3.amazonaws.com/tripdata/202302-citibike-tripdata.csv.zip','https://s3.amazonaws.com/tripdata/202303-citibike-tripdata.csv.zip'],
                        'save_path' : ['./data/202302-citibike-tripdata.csv.zip','./data/202303-citibike-tripdata.csv.zip'],
                        'read_file' : ['202302-citibike-tripdata.csv','202303-citibike-tripdata.csv'],
                        'out_file' : ['./data/202302-citibike.csv','./data/202303-citibike.csv']
    
                    })


            
   
# Download and Processing Bike Files
def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)



def read_local_file(save_path, read_file):
    with zipfile.ZipFile(save_path) as z:
    
        # open the csv file in the dataset
       with z.open(read_file) as f:
          df = pd.read_csv(f, parse_dates=['started_at','ended_at'], 
                           dtype = {'start_station_id': str,'end_station_id': str,})
            
    z.close()
    return df



def process_location_data(df):
    
    fltr_df = df[(df['start_station_id'].isin(station_lst))].copy()
    fltr_df = fltr_df[['start_station_id','start_station_name','start_lat','start_lng']].drop_duplicates('start_station_id')
    fltr_df.rename({'start_station_id': 'station_id', 
                    'start_station_name': 'station_name',
                    'start_lat': 'lat',
                    'start_lng': 'lng'}, axis=1, inplace=True)
    fltr_df.reset_index()
    
    return fltr_df





def process_bike_out_data(df):
    start_df = df[(df['start_station_id'].isin(station_lst))].copy()
    start_df['bikes_out'] = 0
    
    start_df['started_date'] = start_df['started_at'].dt.floor('H')
    start_df['ended_date'] = start_df['ended_at'].dt.floor('H')    
    
    
    start_df = start_df.groupby(['start_station_id','started_date'])[['bikes_out']].count().reset_index()    
    start_df.rename({'start_station_id': 'station_id', 
                    'started_date': 'date'}, axis=1, inplace=True)
    
    return start_df
    




    
def process_bike_in_data(df):
    end_df = df[(df['end_station_id'].isin(station_lst))].copy()
    end_df['bikes_in'] = 0
    
    
    end_df['started_date'] = end_df['started_at'].dt.floor('H')
    end_df['ended_date'] = end_df['ended_at'].dt.floor('H')


    end_df = end_df.groupby(['end_station_id','ended_date'])[['bikes_in']].count().reset_index()
    
    end_df.rename({'end_station_id': 'station_id', 
                    'ended_date': 'date'}, axis=1, inplace=True)

    return end_df




def merge_in_out_data(start_df, end_df, fltr_df):
    bike_df = pd.merge(start_df, end_df, how="outer", on = ["station_id","date"])
    bike_df = pd.merge(bike_df, fltr_df, how="right", on = ["station_id"])
    
    bike_df.fillna(0,inplace=True)
    bike_df['bikes_chng'] = bike_df['bikes_in'] - bike_df['bikes_out'] 
    
    return bike_df




def write_bike_data(bike_df, file_name):
    bike_df.to_csv(file_name)
    




         
      
      
      
if __name__ == "__main__":
    
    
    df_lst = []
    
    
    for index,row in config_df.iterrows():
        download_url(row['file_url'],row['save_path'])
        df = read_local_file(row['save_path'],row['read_file'])
        
        # build dataframes
        fltr_df = process_location_data(df)
        start_df = process_bike_out_data(df)
        end_df = process_bike_in_data(df)
        
        bike_df = merge_in_out_data(start_df, end_df, fltr_df)
        bike_df.to_csv(row['out_file'], index=False)
        
        df_lst.append(bike_df)
        
        
        
    df =   pd.concat(df_lst)
    df['ride_date'] = bike_df['date']
    df = df.set_index('date')
    df.sort_index(inplace=True)
    df.to_csv(bike_file, index=False)
    
    
    
 
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    