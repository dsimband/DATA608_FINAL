#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 22:07:54 2023

@author: dsimbandumwe
"""



from dash import Dash, dcc, Input, Output, html, ctx
import dash_mantine_components as dmc
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc

import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

import datetime
import math 



app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server





# Functions
def create_table(df):
    columns, values = df.columns, df.values
    header = [html.Tr([html.Th(col) for col in columns])]
    rows = [html.Tr([html.Td(cell) for cell in row]) for row in values]
    table = [html.Thead(header), html.Tbody(rows)]
    return table


def run_arima_model(df,train_per,pred_per):
    
    
    # df = bike_df
    # train_per = 80
    # pred_per = 40
    
    splt_index = round(df.shape[0] * 0.8)
    splt_offset = df.shape[0] - splt_index
    
    
    train_df = df[:splt_index]
    t_idx = round(train_df.shape[0] * (1-train_per/100))
    train_df = train_df[t_idx:].copy()
    #test_df = df[-splt_offset:]
    test_df = df[splt_index:]
    t_idx = round(test_df.shape[0] * pred_per/100)
    test_df = test_df[:t_idx].copy()
    
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
    
    df = pd.concat([train_df,test_df])
        
    return df






# Config 
station_lst = ['7617.07']
bike_file = 'https://raw.githubusercontent.com/dsimband/DATA608_FINAL/main/data/bike.csv'




# Data
bike_df = pd.read_csv(bike_file, parse_dates=['ride_date'],dtype = {'station_id': str})
bike_df = bike_df.sort_values('ride_date', ascending=True)



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
           dcc.Slider(id='train_per', value=100, min=0, max=100, step=20),
        ], style ={'width':'40%','display':'inline-block'} ),  
       
       html.Div([
           html.Div(['Select Prediction Range:'], style={'text-align':'left'}),
           dcc.Slider(id='pred_per', value=100, min=0, max=100, step=20),
        ], style ={'width':'40%', 'display':'inline-block', } ), 
    ],style={'textAlign':'center','marginTop':20,'marginBottom':20,'marginLeft':20,'marginRight':20,'font-size': 10,}),
    
 
   html.Div([
       html.Div([
           html.Div(['Select Model:'], style={'text-align':'left'}),
           html.Div( dcc.Dropdown(options=model_df.set_index('label')['value'].to_dict(), id='model_id', 
                                  value=1, clearable=False),),
        ], style ={'width':'30%','display':'inline-block'} ),  
       
       html.Div([
           html.Div(['Select Station:'], style={'text-align':'left'}),
           html.Div( dcc.Dropdown(options=station_df.set_index('label')['value'].to_dict(), id='station_id',
                                  value='7617.07', clearable=False),),
        ], style ={'width':'30%', 'display':'inline-block', } ),   
    ],style={'textAlign':'center','marginTop':20,'marginBottom':20,'marginLeft':20,'marginRight':20,'font-size': 10,}),   
   
   
   
   html.Div([html.Button('Predict', id='button', n_clicks=0),]),
  
    
    html.Div([
        dmc.Stack(
            children=[
                dmc.Divider(variant="solid"),
                dmc.Divider(variant="dashed"),
                dmc.Divider(variant="dotted"),
            ],style={'textAlign':'center','marginTop':20,'marginBottom':20,'marginLeft':20,'marginRight':20,'font-size': 10,}
        ),
    ]),
    
              
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
    Input("station_id", "value"),
    Input("train_per", "value"),
    Input("pred_per", "value"),
    Input("model_id", "value"),
    Input('button', 'n_clicks'),
)
def graph_update(station_id,train_per,pred_per,model_id,n_clicks):
    
    print(station_id)
    print(train_per)
    print(pred_per)
    print(model_id)
    print(n_clicks)
    
    
    
    #if (math.isnan(n_clicks)) :
    if not "button" == ctx.triggered_id:
        fig = px.line(height=400)
        fig.update_layout(template="simple_white", title="Citi Bike Forecast")
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return None, fig
    
    
    
    # filter
    b_df = bike_df[(bike_df['station_id'] == station_id)].copy()
    
    
    # predictioin
    pred_df = run_arima_model(b_df,train_per,pred_per)
    
    #
    
    
    # create 
    #fig = px.line(bike_df, x='ride_date', y='bikes_chng', height=400) 
    fig = px.line(height=400)
    fig.add_scatter(x=pred_df['ride_date'], y=pred_df['bikes_chng'],
                    marker=dict(size=20, color="lightgray"), name='actual')
    fig.add_scatter(x=pred_df['ride_date'], y=pred_df['bikes_chng_pred'], name='forecast')
    fig.update_layout(template="simple_white", title="Citi Bike Forecast")
    
    
    bike_table = create_table(pred_df)

    
    return bike_table, fig
    
   

    



if __name__ == '__main__': 
    app.run_server()
    
    
    

    
    