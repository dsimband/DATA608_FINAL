#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 22:07:54 2023

@author: dsimbandumwe
"""



from dash import Dash, dcc, Input, Output, html
import dash_mantine_components as dmc
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc



dir_output = './output/trip_map.html'


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server



# Config 

station_lst = ['7617.07']


config_df = pd.DataFrame({
                        'file_url' : ['https://s3.amazonaws.com/tripdata/202302-citibike-tripdata.csv.zip',
                                          'https://s3.amazonaws.com/tripdata/202303-citibike-tripdata.csv.zip'],
                        'save_path' : ['./data/202302-citibike-tripdata.csv.zip',
                                           './data/202303-citibike-tripdata.csv.zip'],
                        'read_file' : ['202302-citibike-tripdata.csv','202303-citibike-tripdata.csv'],
                        'out_file' : ['./data/202302-citibike.csv','./data/202303-citibike.csv']
    
                    })




# Functions
def get_bike_data():
    
    df_lst = []
    
    for index,row in config_df.iterrows():
        df = pd.read_csv(row['out_file'],parse_dates=['date'],dtype = {'station_id': str})
        df_lst.append(df)
     
        
    bike_df =   pd.concat(df_lst)
    bike_df['ride_date'] = bike_df['date']
    bike_df = bike_df.set_index('date')
    bike_df.sort_index(inplace=True)
    
    return bike_df
        
        
    




# Data
bike_df = get_bike_data()




# Dsiplay

app.layout = html.Div(id = 'parent', children = [
    html.H1(id = 'H1', children = 'DATA608 Final Project', style = {'textAlign':'center',
                                            'marginTop':20,'marginBottom':20,
                                            'marginLeft':20,'marginRight':20}),


    html.Div([dcc.Graph(id = 'forcast_plot')] ,style = {'textAlign':'center','marginTop':10,'marginBottom':10}),   
              
             
        
    ],style = {'textAlign':'center','marginTop':20,'marginBottom':20,'marginLeft':20,'marginRight':20,
              'font-size': 10,})
    
    

@app.callback(
    Output("forcast_plot", "figure"),
)
def graph_update(boro_value, tree_value):
    
    fig = px.bar(bike_df, y='bikes_chng', height=800)

    
    return fig
    
   

    



if __name__ == '__main__': 
    app.run_server()
    
    
    
    
    
    
    
    
    
    