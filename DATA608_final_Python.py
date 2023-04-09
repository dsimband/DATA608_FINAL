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
import folium



dir_output = './output/trip_map.html'


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server


m = folium.Map(location=[40.730610, -73.935242],zoom_start=13, tiles = 'CartoDB Positron')
m.save('./output/trip_map.html')




app.layout = html.Div(id = 'parent', children = [
    html.H1(id = 'H1', children = 'DATA608 Module 4', style = {'textAlign':'center',
                                            'marginTop':20,'marginBottom':20,
                                            'marginLeft':20,'marginRight':20}),



        m,
    

        html.Div([dcc.Graph(id = 'bar_plot')] ,style = {'textAlign':'center','marginTop':10,'marginBottom':10}),   
        
        html.Div([]),
               
             
        
    ],style = {'textAlign':'center','marginTop':20,'marginBottom':20,'marginLeft':20,'marginRight':20,
              'font-size': 10,})
    
    

@app.callback(
    Output("tree_table", "children"),
    Output("bar_plot", "figure"),
    Input("transfer-list-boro", "value"),
    Input("transfer-list-trees", "value"),
)
def graph_update(boro_value, tree_value):
   
    
   return None, None
    
   

    



if __name__ == '__main__': 
    app.run_server()
    
    
    
    
    
    
    
    
    
    