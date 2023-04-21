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





app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server



def create_table(df):
    columns, values = df.columns, df.values
    header = [html.Tr([html.Th(col) for col in columns])]
    rows = [html.Tr([html.Td(cell) for cell in row]) for row in values]
    table = [html.Thead(header), html.Tbody(rows)]
    return table


# Config 
station_lst = ['7617.07']
bike_file = 'https://raw.githubusercontent.com/dsimband/DATA608_FINAL/main/data/bike.csv'




# Data
bike_df = pd.read_csv(bike_file, parse_dates=['ride_date'],dtype = {'station_id': str})




# Dsiplay

app.layout = html.Div(id = 'parent', children = [
    html.H1(id = 'H1', children = 'DATA608 Final Project', style = {'textAlign':'center',
                                            'marginTop':20,'marginBottom':20,
                                            'marginLeft':20,'marginRight':20}),


    html.Div([dcc.Graph(id = 'forcast_plot')] ,style = {'textAlign':'center','marginTop':10,'marginBottom':10}),   
    html.Div(["Input: ",dcc.Input(id='input_m', value='initial value m', type='text')]),
              
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
    Input("input_m", "value"),
)
def graph_update(m):
    
    #print(bike_df.shape)
    fig = px.bar(bike_df, y='bikes_chng', height=800)
    fig.update_layout(template="simple_white", title="Bike Demand")
    
    
    bike_table = create_table(bike_df)

    
    return bike_table, fig
    
   

    



if __name__ == '__main__': 
    app.run_server()
    
    
    
    
    
    
    