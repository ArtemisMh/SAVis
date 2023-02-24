# ------------------------------
# 
# Developed by Zeynab (Artemis) Mohseni, 
# Winter 2020
# Last Modification: February 2023
#
# ------------------------------

import dash
import dash_bio as db
from dash import dcc
from dash import html

import dash_daq as daq
from dash.dependencies import Input, Output

import dask.dataframe as dd

import json
import pandas as pd
import numpy as np
import plotly
from plotly import tools
import plotly.express as px
import plotly.graph_objs as go
from dash import dash_table

import seaborn as sns
import plotly.figure_factory as ff

from scipy import stats
from scipy.stats import ttest_1samp

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.utils import shuffle
from sklearn.manifold import TSNE

# import sys; sys.path.append('FIt-SNE')
# from fast_tsne import fast_tsne

from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
from sklearn.svm import LinearSVC

from joblib import Memory

location = './cachedir'
memory = Memory(location, verbose=0)

# -------------------------------  Step 1. Launch the application ------------------------------- 
external_stylesheets = ['/stylesheets.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# ------------------------  Step 2. Import and preprocessing the dataset ------------------------ 
@memory.cache
def import_preprocess(sample_size=10000):     
    df1 = dd.read_csv('data/test.csv', sep=",",low_memory=False, assume_missing=True)

    df1 = df1.compute()
    df1 = df1.sample(n=sample_size)

    df1 = df1.sort_values('Month', ascending=True)
    df1['Month'] = df1['Month'].replace({1: 'Jan.', 2:'Feb.', 3:'Mar.', 4:'Apr.', 5:'May', 6:'Jun.', 7:'Jul.', 8:'Aug.', 9: 'Sept.', 10:'Oct.', 11:'Nov.', 12:'Dec.'})

    T = df1['Student ID'][df1['Student answer'] == 'Correct'].count()
    F = df1['Student ID'][df1['Student answer'] == 'Incorrect'].count()

    df2 = df1.iloc[:,[0,5]]
    df2.set_index('Student ID', inplace=True)

    return df1, df2, T, F
df1, df2, T, F = import_preprocess()

# dropdown options
slice = df1.iloc[:,[0, 4, 7, 8, 9]]
opts = [{'label' : i, 'value' : i} for i in slice]

group_labels  = ['Answer duration']
colors = ['#3c19f0']
axis_labels =  [1,2,3,4,5,6,7,8,9]

# -------------------- university ID and topic ID datasets
#df5.shape -> (6423, 1445)
df5 = pd.read_csv('users_and_chapters.csv', sep=",")
pd.options.display.float_format = '{:,.0f}'.format
df5.set_index('userId', inplace=True)

#df5.shape -> (10000, 1445)
df5 = df5.reindex(df2.index)
df5.fillna(0 , inplace=True)
df5.isna().sum().sum()

#df6.shape -> (13263, 1)
df6 = pd.read_csv('UserUniversity.csv', sep=",")
pd.options.display.float_format = '{:,.0f}'.format
df6.set_index('ID', inplace=True)

#df7.shape -> (10000, 1) (with 'Others')
df7 = df6.reindex(df2.index)
df7.fillna(9 , inplace=True)
df7.isna().sum().sum()

X = df5
Y = df7.values.ravel()

# shuffle data
X, y = shuffle(X, Y, random_state=0)

# ---------------- Methods for creating components in the layout code ----------------
def Card(children, **kwargs):
    return html.Section(children, className="card-style")

def NamedSlider(name, short, min, max, step, val, marks=None):
    if marks:
        step = None
    else:
        marks = {i: i for i in range(min, max + 1, step)}

    return html.Div(
        # Marign: TOP, RIGHT, BOTTOM, LEFT
        style={"margin": "20px 10px 40px 10px"},
        children=[
            f"{name}:",
            html.Div(
                style={"margin-left": "10px"},
                children=[
                    dcc.Slider(
                        id=f"slider-{short}",
                        min=min,
                        max=max,
                        marks=marks,
                        step=step,
                        value=val,
                    )
                ],
            ),
        ],
    )

# ----------------- Answer duration distribution -----------------
sample_means = []
population = df1['Answer duration']

for i in range(100):
    sample = population.sample(100)
    stat1, p1 = ttest_1samp(sample, 0.8)
    sample_means.append(sample.mean())

duration_fig = ff.create_distplot(
    [np.array(sample_means)], 
    group_labels, 
    bin_size=.1, 
    colors=colors
)
duration_fig.update_layout(title_text="Answer duration distribution (M=%.2f, SD=%.2f)" %(sample.mean(),sample.std()), 
    height = 380,
    font=dict(
        size=11,
            )
    )

# ------------------------------- Dashboard ------------------------------- 
app.layout = html.Div([
    # ----------------- Header
    html.Div([
		html.Div([
		    html.H3(children='SAVis: A Learning Analytics Dashboard for Educational Datasets', className='twelve columns'),
		    ], className="row",
		    style = {'padding' : '15px' ,
		             'backgroundColor' : '#000099',
		             'color': 'white'}
			)
	]),

    # ----------------- information
    html.Div([
        html.Div([
            html.Div(
                id="N_Student",
                children=[
                    html.P("No. of students"),
                    daq.LEDDisplay(
                        id="operator-Student",
                        value= len(df1['Student ID'].value_counts()),
                        color="#000099",
                        backgroundColor="#d4ddee",
                        size=30,
                    ),
                ],
                className='onehalf columns',
                style={
                        'textAlign': 'center',
                        'color': 'black',
                        'padding-right': '20px',
                    },
                ),
            html.Div(
                id="N_Topic",
                children=[
                    html.P("No. of topics"),
                    daq.LEDDisplay(
                        id="operator-Topic",
                        value= len(df1['Topic ID'].value_counts()),
                        color="#000099",
                        backgroundColor="#d4ddee",
                        size=30,
                    ),
                ],
                className='onehalf columns',
                style={
                        'textAlign': 'center',
                        'color': 'black',
                        'padding-right': '20px',
                    },
                ),
            html.Div(
                id="N_CoAnswer",
                children=[
                    html.P("No. of correct answers"),
                    daq.LEDDisplay(
                        id="operator-CoAnswer",
                        value= T,
                        color="#000099",
                        backgroundColor="#d4ddee",
                        size=30,
                    ),
                ],
                className='two columns',
                style={
                        'textAlign': 'center',
                        'color': 'black',
                        'padding-right': '20px',
                    },
                ),
            html.Div(
                id="N_IncAnswer",
                children=[
                    html.P("No. of incorrect answers"),
                    daq.LEDDisplay(
                        id="operator-IncAnswer",
                        value= F,
                        color="#000099",
                        backgroundColor="#d4ddee",
                        size=30,
                    ),
                ],
                className='two columns',
                style={
                        'textAlign': 'center',
                        'color': 'black',
                        'padding-right': '20px',
                    },
                ),
            html.Div(
                id="N_Duration_min",
                children=[
                    html.P("Min. duration (min)"),
                    daq.LEDDisplay(
                        id="operator-min_Duration",
                        value= round(df1['Answer duration'].min(), 2),
                        color="#000099",
                        backgroundColor="#d4ddee",
                        size=30,
                    ),
                ],
                className='onehalf columns',
                style={
                        'textAlign': 'center',
                        'color': 'black',
                        'padding-right': '20px',
                    },
                ),
            html.Div(
                id="N_Duration_max",
                children=[
                    html.P("Max. duration (min)"),
                    daq.LEDDisplay(
                        id="operator-max_Duration",
                        value= round(df1['Answer duration'].max(),2),
                        color="#000099",
                        backgroundColor="#d4ddee",
                        size=30,
                    ),
                ],
                className='onehalf columns',
                style={
                        'textAlign': 'center',
                        'color': 'black',
                        'padding-right': '20px',
                    },
                ),
            html.Div(
                id="N_Random",
                children=[
                    html.P("No. of random samples"),
                    daq.LEDDisplay(
                        id="operator-led3",
                        value= len(df1),
                        color="#ff0000",
                        backgroundColor="#d4ddee",
                        size=30,
                    ),
                ],
                className='two columns',
                style={
                        'textAlign': 'center',
                        'color': 'black',
                    },
                ),
            ], 
            id="info-container",
            className="row",
            style = {
                    'padding-top': '10px',
                    'padding-down':'10px',
                    'padding-right': '20px',
                    'padding-left':'20px',
                    'backgroundColor' : '#d4ddee',
                    'color': 'black'}
            )
    ]),
    
    # ----------------- T-SNE
    html.Div([ 
        html.Div([
            html.Div([
                html.H6("T-SNE Parameters", id="tsne_h4"),
                html.Div(
                    children=[
                    Card([
                        NamedSlider(
                            # 
                            name="No. of Iterations",
                            short="iterations",
                            min=100,
                            max=500,
                            step=None,
                            val=500,
                            marks={
                                i: str(i) for i in [100, 200, 300, 400, 500]
                            },
                        ),
                        NamedSlider(
                            # number of nearest neighbors that is used in other manifold learning algorithms
                            name="Perplexity",
                            short="perplexity",
                            min=5,
                            max=100,
                            step=None,
                            val=30,
                            marks={i: str(i) for i in [5, 15, 30, 50, 100]},
                        ),
                        NamedSlider(
                            # If the learning rate is too high, the data may look like a ‘ball’ with any point approximately 
                            # equidistant from its nearest neighbours. If the learning rate is too low, most points may look 
                            # compressed in a dense cloud with few outliers.
                            name="Learning Rate",
                            short="learning-rate",
                            min=1,
                            max=200,
                            step=None,
                            val=200,
                            marks={i: str(i) for i in [1, 50, 100, 150, 200]},
                        ),
                        ])
                        ],
                    ),
                ], className= 'two columns',
                style = {
                    'padding-left': '15px',
                    'padding-right':'15px',
                    'height' : 380,
                    'backgroundColor' : 'white',
                    'color': '#3c4861'}
                ),
            html.Div(
                className='fivehalf columns',
                children=[
                    dcc.Loading(id = "loading-icon", 
                            children=[html.Div(dcc.Graph(id='graph-2d-plot-tsne'))], type="circle", color="#2c2c2e"), 
                ],
            style = {
                #'padding-left': '10px',
                #'padding-right': '10px',
                'backgroundColor' : 'white',
                'height' : 380}
            ),

            html.Div(
                className= 'fourhalf columns',
                children=[
                    dcc.Loading(id = "loading-icon0", 
                            children=[html.Div(dcc.Graph(id='predict'))], type="circle", color="#2c2c2e"),       
            ], 
            style = {
                #'padding-left': '10px',
                #'padding-right': '10px',
                'height' : 380,
                'backgroundColor' : 'white',
                'color': '#3c4861'}
            ), 
        ],  
        className="row",
        style = {
            'padding-right': '20px',
            'padding-left':'20px',
            'backgroundColor' : '#d4ddee',
            'color': 'black'}
        )
    ]),

    dcc.Tabs([
        dcc.Tab(label='Student Answer', children=[
            # --------------- scatter plot and pie chart and dropdown
            html.Div([ 
                html.Div(
                    className='eighthalf columns',
                    children=[
                        dcc.Loading(id = "loading-icon1", 
                            children=[html.Div(dcc.Graph(id='cluster1', style={'height':380,'backgroundColor' : 'white'}))], type="circle", color="#2c2c2e"), 
                    ],
                    style = {
                    'backgroundColor' : 'white',
                    'color': '#3c4861'}
                ),
                html.Div(
                    className= 'threehalf columns',
                    children=[
                        html.Div([
                            html.P("Choose a feature:"),
                            dcc.Dropdown(id = 'opt1', options = opts, value = 'Student ID'), 
                            ],
                            style = {
                                'padding-right': '40px',
                                'padding-top': '10px',
                                'padding-left':'20px',
                                'height':83
                                }
                        ),
                        html.Div(
                            children=[
                                dcc.Loading(id = "loading-icon2", 
                                    children=[html.Div(dcc.Graph(id='userAnswerCorrect-pie', style={'height':287,'backgroundColor' : 'white'}))], type="circle", color="#2c2c2e"), 
                            ],
                            style = {
                                'padding-left': '15px',
                                'padding-right': '30px',
                                'backgroundColor' : 'white'}
                        ),            
                    ], 
                
                style = {
                    'padding-left': '15px',
                    'backgroundColor' : 'white',
                    'color': '#3c4861'}
                ), 
            ],  
            className="row",
            style = {
                'padding-right': '20px',
                'padding-left':'20px',
                'backgroundColor' : '#d4ddee',
                'color': 'black'}
            )
        ]),

    	dcc.Tab(label='General Distribution', children=[
            # --------------- main histogram and Radio boxes       
            html.Div([
            	html.Div(
                    className= 'ten columns',
                    children=[
                        dcc.Loading(id = "loading-icon3", 
                            children=[html.Div(dcc.Graph(id='bar1', style={'height':380,'backgroundColor' : 'white'}))], type="circle", color="#2c2c2e"), 
                    ],
                    style = {
                    'backgroundColor' : 'white',
                    'color': '#3c4861'}
                ),
                html.Div(
                    className= 'two columns',
                    children=[
                        html.P("Select a feature:"),
                    	dcc.RadioItems(id = 'items', 
                    		options=[
                            {'label': 'Student ID', 'value': 'Student ID'},
                            {'label': 'Topic ID', 'value': 'Topic ID'},
                            {'label': 'Question type', 'value': 'Question type'},
                            {'label': 'Answer duration', 'value': 'Answer duration'},
                            {'label': 'Monthly activity', 'value': 'Month'},
                            {'label': 'Daily activity', 'value': 'Day'},
                            {'label': 'Hourly activity', 'value': 'Hour'}
                        ], 
                    		value = 'Answer duration'), 
                    ], 
                    style = {
                        'padding-left': '20px',
                        'padding-top': '20px',
                        'padding-right':'20px',
                        'backgroundColor' : 'white',
                        'color': '#3c4861',
                        'height':380}
                    ),
                ],  
            className="row",
            style = {
                'padding-right': '20px',
                'padding-left':'20px',
                'backgroundColor' : '#d4ddee',
                'color': 'black'}
            )
        ]),

        dcc.Tab(label='Answer Duration', children=[
            # --------------- distribution and scatter plots
            html.Div([       
                html.Div([
                    html.Div([
                        dcc.Graph(
                            id='MinuserAnswerDuration-scatter',
                            style={'height':380},
                        )], className= 'seven columns'
                        ),
                    html.Div([
                        dcc.Graph(
                            id='duration-distribution',
                            figure = duration_fig,
                            style={'height':380}
                        )
                        ], className= 'five columns',
                        style = {
                            'backgroundColor' : '#d4ddee',
                            'color': 'black',
                            }
                        )
                    ],  
                className="row",
                style = {
                    'padding-right': '20px',
                    'padding-left':'20px',
                    'backgroundColor' : '#d4ddee',
                    'color': 'black'}
                )
            ])
        ]),

        dcc.Tab(label='Topic Distribution', children=[
            # ---------------  histograms and distribution plots
            html.Div([       
                html.Div([
                    html.Div([
                        dcc.Graph(
                            id='StudentAnswer-histogram',
                            style={'height':380}
                        )], className= 'threehalf columns'
                        ),
                    html.Div([
                        dcc.Graph(
                            id='CorrectAnswer-histogram',
                            style={'height':380}
                        )], className= 'threehalf columns'
                        ),
                    html.Div([
                        dcc.Graph(
                            id='studentanswer-bar',
                            style={'height':380}
                        )], className= 'five columns'
                        )
                    ],  
                className="row",
                style = {
                    'padding-right': '20px',
                    'padding-left':'20px',
                    'backgroundColor' : '#d4ddee',
                    'color': 'black'}
                )
            ])
        ]),

        dcc.Tab(label='Activity', children=[
            # --------------- box and pie plots
            html.Div([       
                html.Div([
                    html.Div([
                        dcc.Graph(
                            id='EventMonth-pie',
                            style={'height':380}
                        )], className= 'three columns'
                        ),
                    html.Div([
                        dcc.Graph(
                            id='questionType-pie',
                            style={'height':380}
                        )], className= 'threehalf columns'
                        ),
                    html.Div([
                        dcc.Graph(
                            id='EventMonth-box',
                            style={'height':380}
                        )], className= 'fivehalf columns',
                        )
                    ],  className="row",
                    style = {
                        'padding-right': '20px',
                        'padding-left':'20px',
                        'backgroundColor' : '#d4ddee',
                        'color': 'black'}
                    )
            ])
        ])   
    ],
    style = {
        'padding-left': '20px',
        'padding-right': '20px',
        'backgroundColor' : '#d4ddee',
        }
    ),

    # --------------- Footer 
    html.Div([
        html.Div([
            html.P('EdTechLnu Group', className='ten columns'),
            html.P('Last updated: May 2022', 
                className='two columns', 
                style={
                    'float': 'right',
                    'position': 'relative'
                    },
            )
        ], className="row",
        style = {
            'padding': '15px',
            'backgroundColor' : '#000099',
            'color': 'white'}
        )
    ])
], className="container")

# ------------------------LEDDisplay definition
def LEDDisplay1(selected_data, column):
    if selected_data:
        indices = [point['pointIndex'] for point in selected_data['points']]
        dff = df1.iloc[indices, :]
    else:
        dff = df1
    return str(len(dff[column].value_counts()))

def LEDDisplay2(selected_data, column):
    if selected_data:
        indices = [point['pointIndex'] for point in selected_data['points']]
        dff = df1.iloc[indices, :]
    else:
        dff = df1
    T = dff['Student ID'][dff[column] == 'Correct'].count()
    return str(T)

def LEDDisplay3(selected_data, column):
    if selected_data:
        indices = [point['pointIndex'] for point in selected_data['points']]
        dff = df1.iloc[indices, :]
    else:
        dff = df1
    F = dff['Student ID'][dff[column] == 'Incorrect'].count()
    return str(F)

def LEDDisplay4(selected_data, column):
    if selected_data:
        indices = [point['pointIndex'] for point in selected_data['points']]
        dff = df1.iloc[indices, :]
    else:
        dff = df1
    Min = round(dff[column].min(), 2)
    return str(Min)

def LEDDisplay5(selected_data, column):
    if selected_data:
        indices = [point['pointIndex'] for point in selected_data['points']]
        dff = df1.iloc[indices, :]
    else:
        dff = df1
    Max = round(dff[column].max(), 2)
    return str(Max)

# ------------------ connecting sliders to TSNE
# @memory.cache
# def run_tsne(iterations, perplexity, learning_rate):
#     tsne_results = fast_tsne(X)
#     return tsne_results

# @app.callback(
#     Output("graph-2d-plot-tsne", "figure"),
#     [
#         Input("slider-iterations", "value"),
#         Input("slider-perplexity", "value"),
#         Input("slider-learning-rate", "value"),
#     ],
# )
# def display_2d_scatter_plot(iterations, perplexity, learning_rate):
#     tsne_results = run_tsne(iterations, perplexity, learning_rate)

#     fig = px.scatter(
#         data_frame = df5,
#         x= tsne_results[:,0],
#         y= tsne_results[:,1],
#         color=y, 
#         height = 380,
#         title = 'T-SNE for student activity in different university IDs',
#         labels ={'x':'', 'y':''},
#         color_continuous_scale='Picnic'
#     ) 
#     fig.update_layout(
#         font=dict(
#                 size=11,
#             )
#         )  
#     return fig

@app.callback(
    Output("graph-2d-plot-tsne", "figure"),
    [
        Input("slider-iterations", "value"),
        Input("slider-perplexity", "value"),
        Input("slider-learning-rate", "value"),
    ],
)
def display_2d_scatter_plot(iterations, perplexity, learning_rate):

    tsne = TSNE(
        n_iter=iterations,
        learning_rate=learning_rate,
        perplexity=perplexity,
        #verbose=999,
    )

    tsne_results = tsne.fit_transform(X) 

    fig = px.scatter(
        data_frame = df5,
        x= tsne_results[:,0],
        y= tsne_results[:,1],
        color=y, 
        height = 380,
        title = 'T-SNE for students\' activity',
        labels ={'x':'', 'y':''},
        color_continuous_scale='Picnic',
    ) 
    fig.update_layout(
        font=dict(
                size=11,
            )
        )  
    return fig

# ------------------------ heatmap
def heatmap1(selected_data, column):
    if selected_data:
        indices = [point['pointIndex'] for point in selected_data['points']]
        dff = df5.iloc[indices, :]
    else:
        dff = df5

    # ----------------------- Prediction -------------------
    #df8.shape (default) -> (6423, 1) (with '9')
    df8 = df6.reindex(dff.index)
    df8.fillna(9, inplace=True)
    df8.isna().sum().sum()
    df8[column] = df8[column].astype(int)
    number = df8[column].value_counts().max()
    X1 = dff
    Y1 = df8.values.ravel()
    X1, y1 = shuffle(X1, Y1, random_state=0)

    # ------------------------- SMOTE ----------------------
    # increace the rows of first table from 6423 to 30258 (X1 is new df)(default) --> because of class 9 which has 3362 values: 3362 * 9
    X1, y1 = SMOTE().fit_resample(X1, y1)
    # increace the rows of second table from 6423 to 30258 (X2 is new df1) (default)--> because of class 9 which has 3362 values: 3362 * 9
    X2 = df8
    X2, y1 = shuffle(X2, Y1, random_state=0)
    X2, y1 = SMOTE().fit_resample(X2, y1)

    model = RandomForestClassifier(n_estimators=100)
    y_pred = cross_val_predict(model, X1, y1, cv=10)
    # calculate accuracy
    accuracy = accuracy_score(y1, y_pred)
    # confusion matrix
    conf_mx = confusion_matrix(y1, y_pred)
    conf_mx1 = (conf_mx/number*100).round()
    fig = ff.create_annotated_heatmap(conf_mx1, colorscale='Picnic', x=axis_labels, y=axis_labels, showscale= True)
    fig.update_layout(title_text="Performance of the RF classifier = %.2f" %(accuracy*100) +"%", 
        font=dict(
            size=11,
        ),
        yaxis_title = "University ID", 
        height = 380,
        xaxis=dict(side="bottom",))  
    return fig


# ------------------ Dropdown to scatter plot - 1st tab: student answer
@app.callback(Output('cluster1', 'figure'),
             [Input('opt1', 'value'), Input('graph-2d-plot-tsne', 'selectedData')])

def update_image_src1(input1, input2):
    if input2:
        indices = [point['pointIndex'] for point in input2['points']]
        dff = df1.iloc[indices, :]
    else:
        dff = df1

    dff['Student ID'] = dff['Student ID'].astype(int)
    #dff['Student ID'] = dff['Student ID'].astype(str)
    dff1 = pd.DataFrame(columns=[])
    dff1['Count']=dff['Student ID'].value_counts()
    dff1['U_C'] = dff['Student ID'][dff['Student answer'] == 'Correct'].value_counts()
    dff1['U_I'] = dff['Student ID'][dff['Student answer'] == 'Incorrect'].value_counts()
    dff1['User']=dff['Student ID'].value_counts().index
    dff1.fillna(0 , inplace=True)

    if input1 == 'Student ID':
        dff1 = dff1.sort_values('User', ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            name="Correct",
            mode="markers+lines", x=dff1.User, y=dff1['U_C'],
            #xperiod="M1",
            xperiodalignment="middle",
            connectgaps=True
        ))
        fig.add_trace(go.Scatter(
            name="Incorrect",
            mode="markers+lines", x=dff1.User, y=dff1['U_I'],
            #xperiod="M1",
            xperiodalignment="middle",
            connectgaps=True
        ))
        fig.update_layout(
            title="Number of students\' answers",
            xaxis_title="Student IDs",
            yaxis_title="Count",
            height = 380,
            hovermode="x unified",
            font=dict(
                size=11,
            ),
        )
        fig.update_traces(
            hoverinfo="y+x+name",
            #showlegend=True,

        )
        return fig

    dff['Topic ID'] = dff['Topic ID'].astype(int)
    dff1 = pd.DataFrame(columns=[])
    dff1['Count']=dff['Topic ID'].value_counts()
    dff1['T_C'] = dff['Topic ID'][dff['Student answer'] == 'Correct'].value_counts()
    dff1['T_I'] = dff['Topic ID'][dff['Student answer'] == 'Incorrect'].value_counts()
    dff1['Topic']=dff['Topic ID'].value_counts().index
    dff1.fillna(0 , inplace=True)

    if input1 == 'Topic ID':
        dff1 = dff1.sort_values('Topic', ascending=False)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            name="Correct",
            mode="markers+lines", x=dff1.Topic, y=dff1['T_C'],
            #xperiod="M1",
            xperiodalignment="middle",
            connectgaps=True
        ))
        fig.add_trace(go.Scatter(
            name="Incorrect",
            mode="markers+lines", x=dff1.Topic, y=dff1['T_I'],
            #xperiod="M1",
            xperiodalignment="middle",
            connectgaps=True
        ))
        fig.update_layout(
            title="Number of students\' answers according to topic ID",
            xaxis_title="Topic IDs",
            yaxis_title="Count",
            height = 380,
            hovermode="x unified",
            font=dict(
                size=11,
            ),
        )
        fig.update_traces(
            hoverinfo="y+x+name",
            #showlegend=True,

        )
        return fig

    dff1 = pd.DataFrame(columns=[])
    dff1['Count']=dff['Month'].value_counts()
    dff1['M_C'] = dff['Month'][dff['Student answer'] == 'Correct'].value_counts()
    dff1['M_I'] = dff['Month'][dff['Student answer'] == 'Incorrect'].value_counts()
    dff1['Monthly']=dff['Month'].value_counts().index
    dff1.fillna(0 , inplace=True)

    if input1 == 'Month':
        dff1 = dff1.sort_values('Monthly', ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            name="Correct",
            mode="markers+lines", x=dff1.Monthly, y=dff1['M_C'],
            #xperiod="M1",
            xperiodalignment="middle",
            connectgaps=True
        ))
        fig.add_trace(go.Scatter(
            name="Incorrect",
            mode="markers+lines", x=dff1.Monthly, y=dff1['M_I'],
            #xperiod="M1",
            xperiodalignment="middle",
            connectgaps=True
        ))
        fig.update_layout(
            title="Number of students\' answers per month",
            xaxis_title="Month",
            yaxis_title="Count",
            height = 380,
            hovermode="x unified",
            font=dict(
                size=11,
            ),
        )
        fig.update_traces(
            hoverinfo="y+x+name",
            #showlegend=True,

        )
        return fig

    dff1 = pd.DataFrame(columns=[])
    dff1['Count']=dff['Day'].value_counts()
    dff1['D_C'] = dff['Day'][dff['Student answer'] == 'Correct'].value_counts()
    dff1['D_I'] = dff['Day'][dff['Student answer'] == 'Incorrect'].value_counts()
    dff1['Daily']=dff['Day'].value_counts().index
    dff1.fillna(0 , inplace=True)

    if input1 == 'Day':
        dff1 = dff1.sort_values('Daily', ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            name="Correct",
            mode="markers+lines", x=dff1.Daily, y=dff1['D_C'],
            #xperiod="M1",
            xperiodalignment="middle",
            connectgaps=True
        ))
        fig.add_trace(go.Scatter(
            name="Incorrect",
            mode="markers+lines", x=dff1.Daily, y=dff1['D_I'],
            #xperiod="M1",
            xperiodalignment="middle",
            connectgaps=True
        ))
        fig.update_layout(
            title="Number of students\' answers per day",
            xaxis_title="Day",
            yaxis_title="Count",
            height = 380,
            hovermode="x unified",
            font=dict(
                size=11,
            ),
        )
        fig.update_traces(
            hoverinfo="y+x+name",
            #showlegend=True,

        )
        return fig

    dff1 = pd.DataFrame(columns=[])
    dff1['Count']=dff['Hour'].value_counts()
    dff1['H_C'] = dff['Hour'][dff['Student answer'] == 'Correct'].value_counts()
    dff1['H_I'] = dff['Hour'][dff['Student answer'] == 'Incorrect'].value_counts()
    dff1['Hourly']=dff['Hour'].value_counts().index
    dff1.fillna(0 , inplace=True)

    if input1 == 'Hour':
        dff1 = dff1.sort_values('Hourly', ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            name="Correct",
            mode="markers+lines", x=dff1.Hourly, y=dff1['H_C'],
            #xperiod="M1",
            xperiodalignment="middle",
            connectgaps=True
        ))
        fig.add_trace(go.Scatter(
            name="Incorrect",
            mode="markers+lines", x=dff1.Hourly, y=dff1['H_I'],
            #xperiod="M1",
            xperiodalignment="middle",
            connectgaps=True
        ))
        fig.update_layout(
            title="Number of students\' answers per hour",
            xaxis_title="Hour",
            yaxis_title="Count",
            height = 380,
            hovermode="x unified",
            font=dict(
                size=11,
            ),
        )
        fig.update_traces(
            hoverinfo="y+x+name",
            #showlegend=True,

        )
        return fig

# ------------------------ pie chart - 1st tab: student answer
def pie1(selected_data, column):
    if selected_data:
        indices = [point['pointIndex'] for point in selected_data['points']]
        dff = df1.iloc[indices, :]
    else:
        dff = df1
    fig = px.pie(
        data_frame = dff,
        names = column,
        hole = .3,
        height = 287,
        title = 'Students\' answers (%)',
    )
    fig.update_layout(
        font=dict(
            size=11,
        )
    )
    return fig

# ------------------ Dropdown to Bar plot - 2nd tab: general distribution
@app.callback(Output('bar1', 'figure'),
             [Input('items', 'value'), Input('graph-2d-plot-tsne', 'selectedData')])

def update_image_src2(input1, input2):
    if input2:
        indices = [point['pointIndex'] for point in input2['points']]
        dff = df1.iloc[indices, :]
    else:
        dff = df1

    dff['Student ID'] = dff['Student ID'].astype(int)

    if  input1 == 'Month':
        fig = px.histogram(
            data_frame = dff,
            x="Month",
            color="Student answer", 
            marginal="rug", 
            hover_data=dff.columns,
            labels ={'Monthly activity':'Month', 'y':'number of user intractions'},
            title = 'Distribution of studnets monthly activity',
            height = 380
        )
        fig.update_layout(
            font=dict(
                size=11,
            )
        )

    if  input1 == 'Day':
        fig = px.histogram(
            data_frame = dff,
            x="Day",
            color="Student answer", 
            marginal="rug", 
            hover_data=dff.columns,
            labels ={'Daily activity':'Day'},
            title = 'Distribution of studnets daily activity',
            height = 380
        )
        fig.update_layout(
            font=dict(
                size=11,
            )
        )

    if  input1 == 'Hour':
        fig = px.histogram(
            data_frame = dff,
            x="Hour",
            color="Student answer", 
            marginal="rug", 
            hover_data=dff.columns,
            labels ={'Hourly activity':'Hour'},
            title = 'Distribution of studnets hourly activity',
            height = 380
        )
        fig.update_layout(
            font=dict(
                size=11,
            )
        )

    if  input1 == 'Answer duration':
        fig = px.histogram(
            data_frame = dff,
            x="Answer duration",
            color="Student answer", 
            marginal="rug", 
            hover_data=dff.columns,
            labels ={'Answer duration':'Answer duration'},
            title = 'Distribution of students\' answer duration',
            height = 380
        )
        fig.update_layout(
            font=dict(
                size=11,
            )
        )

    if  input1 == 'Topic ID':
        fig = px.histogram(
            data_frame = dff,
            x="Topic ID",
            color="Student answer", 
            marginal="rug", 
            hover_data=dff.columns,
            labels ={'Topic ID':'Topic ID'},
            title = 'Distribution of topic ID',
            height = 380
        )
        fig.update_layout(
            font=dict(
                size=11,
            )
        )
  
    if  input1 == 'Student ID':
        fig = px.histogram(
            data_frame = dff,
            x="Student ID",
            color="Student answer", 
            marginal="rug", 
            hover_data=dff.columns,
            labels ={'Student ID':'Student ID'},
            title = 'Distribution of student ID',
            height = 380
        )
        fig.update_layout(
            font=dict(
                size=11,
            )
        )

    if  input1 == 'Question type':
        fig = px.histogram(
            data_frame = dff,
            x="Question type",
            color="Student answer", 
            marginal="rug", 
            labels ={'Question type':'Question type'},
            title = 'Distribution of question type',
            hover_data=dff.columns,
            height = 380
        )   
        fig.update_layout(
            font=dict(
                size=11,
            )
        ) 
    return fig

# ------------------------ Scatter plots - 3rd tab: Answer duration
def scatter1(selected_data, column):
    if selected_data:
        indices = [point['pointIndex'] for point in selected_data['points']]
        dff = df1.iloc[indices, :]
    else:
        dff = df1

    fig = px.scatter(
        data_frame = dff,
        x="Answer duration", 
        y="Topic ID", 
        animation_frame="Month", 
        animation_group="Resource name",
        size="Answer duration", 
        color="Student answer", 
        hover_name="Student ID",
        facet_col="Student answer",
        log_x=True, 
        size_max=35, 
        range_x=[0.01,20], 
        range_y=[600,3501],
        height = 380,
        title = 'Students\' answer duration',
        labels ={'Answer duration':'Students\' answer duration (min)', 'Topic ID':'Topic ID'}
    )   
    fig.update_layout(
        font=dict(
            size=11,
        )
    )
    return fig

# ------------------------histogram - topic distribution for student answers - 4th tab: topic distribution
def histogram1(selected_data, column):
    if selected_data:
        indices = [point['pointIndex'] for point in selected_data['points']]
        dff = df1.iloc[indices, :]
    else:
        dff = df1

    
    fig = px.histogram(
            data_frame = dff,
            x="Topic ID",
            color= column,
            nbins = 30,
            title = 'Distribution of students\' answers',
            labels ={'Topic ID':'Topic IDs', 'Student answer':'student answers'},
            height = 380
        ) 
    fig.update_layout(
        font=dict(
            size=11,
        )
    )
    return fig

# ------------------------histogram - topic distribution for correct answers - 4th tab: topic distribution
def histogram2(selected_data, column):
    if selected_data:
        indices = [point['pointIndex'] for point in selected_data['points']]
        dff = df1.iloc[indices, :]
    else:
        dff = df1

    dff1 = dff[(dff[column] == 'Correct')]
    fig = px.histogram(
            data_frame = dff1,
            x= "Topic ID",
            color= column,
            nbins = 30,
            title = 'Distribution of correct answers',
            labels ={'Topic ID':'Topic IDs'},
            height = 380
        ) 
    fig.update_layout(
        font=dict(
            size=11,
        )
    )
    return fig

# ------------------------bar plot top 10 definition - 4th tab: topic distribution
def bar2(selected_data, column):
    if selected_data:
        indices = [point['pointIndex'] for point in selected_data['points']]
        dff = df1.iloc[indices, :]
    else:
        dff = df1

    trace_1 = go.Scatter(
                        x = sorted(dff['Resource name'][dff['Student answer'] == 'Correct'].value_counts().head(10).index), 
                        y = dff['Resource name'][dff['Student answer'] == 'Correct'].value_counts().head(10), 
                        mode='lines+markers',
                        name = 'Correct',
                        opacity=0.8,
                        marker={
                            'size': 9})
    trace_2 = go.Scatter(
                        x = sorted(dff['Resource name'][dff['Student answer'] == 'Correct'].value_counts().head(10).index), 
                        y = dff['Resource name'][dff['Student answer'] == 'Incorrect'].value_counts().head(10), 
                        mode='lines+markers',
                        name = 'Incorrect',
                        opacity=0.8,
                        marker={
                            'size': 9})

    layout = go.Layout(title = 'Top 10 topics', 
        hovermode = 'closest', 
        height= 380, 
        xaxis_title="",
        font=dict(
            size=11,
        ),
        yaxis_title="No. of students\' answers")
    fig = go.Figure(data = [trace_1, trace_2], layout = layout)
    return fig

# ------------------------pie plot-monthly activity definition - 5th tab: Activity
def pie2(selected_data, column):
    if selected_data:
        indices = [point['pointIndex'] for point in selected_data['points']]
        dff = df1.iloc[indices, :]
    else:
        dff = df1
    #dff['Month'] = dff['Month'].replace({1: 'Jan.', 2:'Feb.', 3:'Mar.', 4:'Apr.', 5:'May', 6:'Jun.', 7:'Jul.', 
    #    8:'Aug.', 9: 'Sept.', 10:'Oct.', 11:'Nov.', 12:'Dec.'})
    
    fig = px.pie(
        data_frame = dff,
        names = column,
        hole = .3,
        height = 380,
        title = 'Students\' activity per month (%)',
    )
    fig.update_layout(
        font=dict(
            size=11,
        )
    )
    return fig

# ------------------------pie questionType definition - 5th tab: Activity
def pie3(selected_data, column):
    if selected_data:
        indices = [point['pointIndex'] for point in selected_data['points']]
        dff = df1.iloc[indices, :]
    else:
        dff = df1
    fig = px.pie(
        data_frame = dff,
        names = column,
        hole = .3,
        height = 380,
        title = 'Question types (%)',
    )
    fig.update_layout(
        font=dict(
            size=11,
        )
    )
    return fig

 # ------------------------box plot_Monthly acitvity definition - 5th tab: Activity
def box(selected_data, column):
    if selected_data:
        indices = [point['pointIndex'] for point in selected_data['points']]
        dff = df1.iloc[indices, :]
    else:
        dff = df1


    fig = px.box(
        data_frame = dff,
        x = dff[column],
        y = dff['Topic ID'],
        color= dff['Student answer'],
        height = 380,
        title= 'Distribution of students\' activity per month',
        labels ={'Monthly activity':'Students\' activity per month', 'Topic ID':'Topic IDs'},
    )
    fig.update_layout(
        font=dict(
            size=11,
        )
    )
    return fig

# ------------------ T-SNE to LEDDisplays
@app.callback(
    Output('operator-Student', 'value'),
    [Input('graph-2d-plot-tsne', 'selectedData')])
def update_N_Student(selected_data):
    return LEDDisplay1(selected_data, 'Student ID')

@app.callback(
    Output('operator-Topic', 'value'),
    [Input('graph-2d-plot-tsne', 'selectedData')])
def update_N_Topic(selected_data):
    return LEDDisplay1(selected_data, 'Topic ID')

@app.callback(
    Output('operator-CoAnswer', 'value'),
    [Input('graph-2d-plot-tsne', 'selectedData')])
def update_N_CoAnswer(selected_data):
    return LEDDisplay2(selected_data, 'Student answer')

@app.callback(
    Output('operator-IncAnswer', 'value'),
    [Input('graph-2d-plot-tsne', 'selectedData')])
def update_N_IncAnswer(selected_data):
    return LEDDisplay3(selected_data, 'Student answer')

@app.callback(
    Output('operator-min_Duration', 'value'),
    [Input('graph-2d-plot-tsne', 'selectedData')])
def update_N_min(selected_data):
    return LEDDisplay4(selected_data, 'Answer duration')

@app.callback(
    Output('operator-max_Duration', 'value'),
    [Input('graph-2d-plot-tsne', 'selectedData')])
def update_N_max(selected_data):
    return LEDDisplay5(selected_data, 'Answer duration')

# ------------------ T-SNE to heatmap
@app.callback(
    Output('predict', 'figure'),
    [Input('graph-2d-plot-tsne', 'selectedData')])
def update_userAnswerCorrect(selected_data):
    return heatmap1(selected_data, 'UniversityID')

# ------------------ T-SNE to pie chart - 1st tab: student answer
@app.callback(
    Output('userAnswerCorrect-pie', 'figure'),
    [Input('graph-2d-plot-tsne', 'selectedData')])
def update_userAnswerCorrect(selected_data):
    return pie1(selected_data, 'Student answer')


# ------------------ T-SNE to Scatter plot - 3rd tab: Answer duration
@app.callback(
    Output('MinuserAnswerDuration-scatter', 'figure'),
    [Input('graph-2d-plot-tsne', 'selectedData')])
def update_MinuserAnswerDuration(selected_data):
    return scatter1(selected_data, 'Answer duration')

# ------------------ T-SNE to histograms and bar- 4th tab: topic distribution
@app.callback(
    Output('StudentAnswer-histogram', 'figure'),
    [Input('graph-2d-plot-tsne', 'selectedData')])
def update_Topic_StudentAnswer(selected_data):
    return histogram1(selected_data, 'Student answer')

@app.callback(
    Output('CorrectAnswer-histogram', 'figure'),
    [Input('graph-2d-plot-tsne', 'selectedData')])
def update_Topic_CorrectAnswer(selected_data):
    return histogram2(selected_data, 'Student answer')

@app.callback(
    Output('studentanswer-bar', 'figure'),
    [Input('graph-2d-plot-tsne', 'selectedData')])
def update_top10_StudentAnswer(selected_data):
    return bar2(selected_data, 'Topic ID')

# ------------------ T-SNE to pie charts and box plot- 5th tab: Activity
@app.callback(
    Output('EventMonth-pie', 'figure'),
    [Input('graph-2d-plot-tsne', 'selectedData')])
def update_EventMonth_pie(selected_data):
    return pie2(selected_data, 'Month')

@app.callback(
    Output('questionType-pie', 'figure'),
    [Input('graph-2d-plot-tsne', 'selectedData')])
def update_QT(selected_data):
    return pie3(selected_data, 'Question type')

@app.callback(
    Output('EventMonth-box', 'figure'),
    [Input('graph-2d-plot-tsne', 'selectedData')])
def update_EventMonth(selected_data):
    return box(selected_data, 'Month')

if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port='8080')
