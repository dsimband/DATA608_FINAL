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

import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

import datetime




app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server





# Functions
def create_table(df):
    columns, values = df.columns, df.values
    header = [html.Tr([html.Th(col) for col in columns])]
    rows = [html.Tr([html.Td(cell) for cell in row]) for row in values]
    table = [html.Thead(header), html.Tbody(rows)]
    return table


def run_arima_model(df):
    
    
    #df = bike_df
    
    splt_index = round(df.shape[0] * 0.8)
    splt_offset = df.shape[0] - splt_index
    
    
    train_df = df[:splt_index]
    #test_df = df[-splt_offset:]
    test_df = df[splt_index:]
    
    arima_model = pm.auto_arima(train_df[['bikes_chng']], test='adf', 
                             #max_p=3, max_d=3, max_q=3, 
                             #max_P=3, max_D=2, max_Q=3,
                             seasonal=True, 
                             m=1,
                             trace=True,
                             error_action='ignore',  
                             suppress_warnings=True, 
                             stepwise=True)
    
    pred_df = pd.DataFrame(arima_model.predict(n_periods = test_df.shape[0]),index=test_df.index)
    pred_df.columns = ['bikes_chng_pred']
    
    
    #pred_df['ride_date'] = test_df['ride_date']
    test_df['bikes_chng_pred'] = pred_df['bikes_chng_pred']
        
    return test_df






# Config 
station_lst = ['7617.07']
bike_file = 'https://raw.githubusercontent.com/dsimband/DATA608_FINAL/main/data/bike.csv'




# Data
bike_df = pd.read_csv(bike_file, parse_dates=['ride_date'],dtype = {'station_id': str})
bike_df = bike_df.sort_values('ride_date', ascending=True)

# s_date = bike_df.copy().iloc[-1]['ride_date']
# e_date = s_date + datetime.timedelta(days=30)
# step_date = (e_date - s_date) / 10


model_df = pd.DataFrame({'label':[1],
                         'value':'ARIMA'})

station_df = bike_df[['station_id','station_name']].drop_duplicates(keep='first')
station_df.rename(columns = {'station_id':'label',
                             'station_name':'value'}, inplace = True)


s = model_df.set_index('label')['value'].to_dict()


# Dsiplay
app.layout = html.Div(id = 'parent', children = [
    html.H1(id = 'H1', children = 'DATA608 Final Project', style = {'textAlign':'center',
                                            'marginTop':20,'marginBottom':20,
                                            'marginLeft':20,'marginRight':20}),


    html.Div([
        dmc.LoadingOverlay(
            html.Div([dcc.Graph(id = 'forcast_plot')] ,
                     style = {'textAlign':'center','marginTop':10,'marginBottom':10}),
        ),
    ]),  
    
    
    
    
   
   html.Div([
       html.Div([
           html.Div(['Select % of Training Data To Use:'], style={'text-align':'left'}),
           dcc.Slider(id='my-slider', value=100, min=0, max=100, step=20),
        ], style ={'width':'40%','display':'inline-block'} ),  
       
       html.Div([
           html.Div(['Select Prediction Range:'], style={'text-align':'left'}),
           dcc.Slider(id='my-slider2', value=100, min=0, max=100, step=20),
        ], style ={'width':'40%', 'display':'inline-block', } ), 
    ],style={'textAlign':'center','marginTop':20,'marginBottom':20,'marginLeft':20,'marginRight':20,'font-size': 10,}),
    
 
   html.Div([
       html.Div([
           html.Div(['Select Model:'], style={'text-align':'left'}),
           #html.Div( dcc.Dropdown(['NYC', 'MTL', 'SF'], 'NYC', id='model_id'),),
           html.Div( dcc.Dropdown(options=model_df.set_index('label')['value'].to_dict(), id='model_id'),),
        ], style ={'width':'30%','display':'inline-block'} ),  
       
       html.Div([
           html.Div(['Select Station:'], style={'text-align':'left'}),
           html.Div( dcc.Dropdown(options=station_df.set_index('label')['value'].to_dict(), id='station_id'),),
        ], style ={'width':'30%', 'display':'inline-block', } ),   
    ],style={'textAlign':'center','marginTop':20,'marginBottom':20,'marginLeft':20,'marginRight':20,'font-size': 10,}),   
   
   
   
   html.Div([html.Button('Predict', id='button'),]),
  
   
    
              
    html.Div([
        html.Div([
            dmc.Table(
                id = 'bike_table',
                striped=True,
                highlightOnHover=True,
                withBorder=False,
                withColumnBorders=True,
        ),], style={'width': '95%', 'float': 'center', 'display': 'inline-block','textAlign':'center'}),
    ]),  
        
    ],style = {'textAlign':'center','marginTop':20,'marginBottom':20,'marginLeft':20,'marginRight':20,
              'font-size': 10,})
    
    

@app.callback(
    Output("bike_table", "children"),
    Output("forcast_plot", "figure"),
    Input("model_id", "value"),
)
def graph_update(m):
    
    
    pred_df = run_arima_model(bike_df)
    
    #print(bike_df.shape)
    #fig = px.bar(bike_df, y='bikes_chng', height=800)
    #fig.update_layout(template="simple_white", title="Bike Demand")
    
    #fig = px.line(bike_df, x='ride_date', y='bikes_chng', height=800) 
    #fig.add_scatter(x=pred_df['ride_date'], y=pred_df['bikes_chng'])
    
    fig = px.line(bike_df, x='ride_date', y='bikes_chng', height=800) 
    fig.add_scatter(x=pred_df['ride_date'], y=pred_df['bikes_chng_pred'])
    
    
    bike_table = create_table(pred_df)

    
    return bike_table, fig
    
   

    



if __name__ == '__main__': 
    app.run_server()
    
    
    
    
    
    
    