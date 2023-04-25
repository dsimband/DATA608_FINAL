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
from statsmodels.tsa.api import Holt, ExponentialSmoothing
from neuralprophet import NeuralProphet



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





def process_bike_table(df):
    
    c_df = df.copy()   
    c_df['ride_date'] = c_df['ride_date'].dt.floor('D')   
    c_df = c_df.groupby(['station_id','station_name','ride_date'])[['bikes_out','bikes_in','bikes_chng','bikes_chng_pred']].sum().reset_index()  
    c_df = c_df[['station_id','ride_date','station_name','bikes_out','bikes_in','bikes_chng','bikes_chng_pred']] 
    c_df = c_df.dropna()
        
    return c_df




 
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
                              m=7,
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




def run_ets_model(df, train_per, pred_per):
    
   #df = bike_df[(bike_df['station_id'] == '7617.07')].copy()
   #train_per = 80
   #pred_per = 40
   
   splt_index = round(df.shape[0] * 0.8)
   splt_offset = df.shape[0] - splt_index
   
   
   train_df = df[:splt_index]
   t_idx = round(train_df.shape[0] * (1-train_per/100))
   train_df = train_df[t_idx:].copy()
   train = train_df[['bikes_chng','ride_date']]
   train.set_index('ride_date', inplace=True)
   train = train.resample('1H').sum()
   
   
   
   #test_df = df[-splt_offset:]
   test_df = df[splt_index:]
   t_idx = round(test_df.shape[0] * pred_per/100)
   test_df = test_df[:t_idx].copy()
   test = test_df[['bikes_chng','ride_date']]
   test.set_index('ride_date', inplace=True)
   test = test.resample('1H').sum()

   hwm = ExponentialSmoothing(train, seasonal_periods=24*7, 
                              trend='add', 
                              seasonal='add', 
                              use_boxcox=False, 
                              initialization_method='estimated').fit()
   
   hwm_pred = hwm.forecast(len(test))

   hwm_df = pd.DataFrame(hwm_pred)
   hwm_df.columns = ['bikes_chng_pred']
   hwm_df.reset_index(inplace=True)
   hwm_df.rename({'index': 'ride_date'}, axis=1, inplace=True)

   df = pd.concat([train_df,test_df])
   df = pd.merge(df, hwm_df, how="outer", on = ["ride_date"])
   df.sort_values('ride_date', inplace=True)
   
   return df
    


def run_prophet_model(df, train_per, pred_per):
    
   # df = bike_df[(bike_df['station_id'] == '7617.07')].copy()
   # train_per = 80
   # pred_per = 40
   
   splt_index = round(df.shape[0] * 0.8)
   splt_offset = df.shape[0] - splt_index
   
  
   train_df = df[:splt_index]
   t_idx = round(train_df.shape[0] * (1-train_per/100))
   train_df = train_df[t_idx:]
   train = train_df[['bikes_chng','ride_date']].copy()
   train.rename({'ride_date' : 'ds' , 'bikes_chng' :'y'
       }, axis=1, inplace=True)
   train.set_index('ds', inplace=True)
   train = train.resample('1H').sum()   
   train.reset_index(inplace=True)

   test_df = df[splt_index:]
   t_idx = round(test_df.shape[0] * pred_per/100)
   test_df = test_df[:t_idx]
   test = test_df[['bikes_chng','ride_date']].copy()
   test.rename({'ride_date' : 'ds' , 'bikes_chng' :'y'
       }, axis=1, inplace=True)
   test.set_index('ds', inplace=True)
   test = test.resample('1H').sum()
   test.reset_index(inplace=True)


   # t_df = df[['ride_date','bikes_chng']].copy()
   # t_df.set_index('ride_date', inplace=True)
   # t_df = t_df.resample('1H').sum()
   # t_df.reset_index(inplace=True)
   # t_df.rename({'bikes_chng': 'y',
   #                  'ride_date': 'ds'}, axis=1, inplace=True)

   # train = t_df[:splt_index]
   # test = t_df[splt_index:]
   
   
   # Prophet Forecast
   m = NeuralProphet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True
    )
   
   metrics = m.fit(train, freq='H', validation_df=test, progress="plot")
   forecast = m.predict(test)
   
  
   proph_pred = forecast[['ds','yhat1']].copy()
   proph_pred.rename({'ds': 'ride_date',
                      'yhat1' : 'bikes_chng_pred'
                      }, axis=1, inplace=True)
   
   df = pd.concat([train_df,test_df])
   df = pd.merge(df, proph_pred, how="outer", on = ["ride_date"])
   df.sort_values('ride_date', inplace=True)
   
   return df








# Config 
station_lst = ['7617.07']
bike_file = 'https://raw.githubusercontent.com/dsimband/DATA608_FINAL/main/data/bike.csv'




# Data
bike_df = pd.read_csv(bike_file, parse_dates=['ride_date'],dtype = {'station_id': str})
bike_df = bike_df.sort_values('ride_date', ascending=True)



model_df = pd.DataFrame({'label':['1','2','3'],
                         'value':['Auto SARIMA', 'NeuralProphet' , 'Exponential Smoothing']})

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
           dcc.Slider(id='train_per', value=100, min=50, max=100, step=10),
        ], style ={'width':'40%','display':'inline-block'} ),  
       
       html.Div([
           html.Div(['Select Prediction Range:'], style={'text-align':'left'}),
           dcc.Slider(id='pred_per', value=100, min=20, max=100, step=20),
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
    print(type(station_id))
    print(train_per)
    print(pred_per)
    print('model_id: ' + str(model_id))
    print(type(model_id))
    print(n_clicks)
    print(ctx.triggered_id)
    
    
    
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
    if model_id == '1':
        pred_df = run_arima_model(b_df,train_per,pred_per)
    elif model_id == '2':
        pred_df = run_prophet_model(b_df,train_per,pred_per)
    else:
        pred_df = run_ets_model(b_df,train_per,pred_per)
    
    
    # create 
    #fig = px.line(bike_df, x='ride_date', y='bikes_chng', height=400) 
    fig = px.line(height=400)
    fig.add_scatter(x=pred_df['ride_date'], y=pred_df['bikes_chng'],
                    marker=dict(size=20, color="lightgray"), name='actual')
    fig.add_scatter(x=pred_df['ride_date'], y=pred_df['bikes_chng_pred'], name='forecast')
    fig.update_layout(template="simple_white", title="Citi Bike Forecast")
    
    
    
    pred_day_df = process_bike_table(pred_df)
    bike_table = create_table(pred_day_df)

    
    return bike_table, fig
    
   

    



if __name__ == '__main__': 
    app.run_server()
    
    



    
    