import dash
import dash_core_components as dcc
import dash_html_components as html
from app import app
import plotly.graph_objects as go
import pandas as pd

sev = dash.Dash(
    __name__,
    server=app,
    routes_pathname_prefix='/model_details/',
    title='Model Evaluation'
)

df = pd.read_csv('data/accuracy.csv')

name = df['Name'].unique()

sev.layout = html.Div([
    html.Div([
        html.H1('Model Evaluation')
    ], style={'display': 'inline-block', 'width': '49%', 'color': 'white'}),
    html.Link(
            rel='stylesheet',
            href='/static/style.css'
    ),
    dcc.Dropdown(
        id='crossfilter-model-column',
        options=[{'label': i, 'value': i} for i in name],
        value='Models'
    ),
    html.Br(),
    dcc.Graph(id='crossfilter-indicator-scatter')
], style={'display': 'inline-block', 'width': '49%', 'color': 'black', 'margin-top': '0px'})

@sev.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),
    [dash.dependencies.Input('crossfilter-model-column', 'value')])
def update_graph(name):
    df1 = df.loc[df['Name'] == name]
    fig = go.Figure(go.Scatter(x = df1['Data Points'], y = df1['Accuracy']))
    fig.update_xaxes(title='Data Points')
    fig.update_yaxes(title='Accuracy')
    fig.update_layout(plot_bgcolor='rgb(230, 230,230)', showlegend=True)
    return fig

if __name__ == '__main__':
    sev.run_server(host='0.0.0.0',debug=False, port=8050)
