import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output


import json
import pandas as pd
import numpy as np
import plotly
from plotly import tools
import plotly.express as px
import plotly.graph_objs as go
import dash_table

import seaborn as sns
import plotly.figure_factory as ff

from scipy import stats
from scipy.stats import ttest_1samp

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
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


# Step 1. Launch the application
external_stylesheets = ['/stylesheets.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Step 2. Import the dataset
df = pd.read_csv('cleaned_answer_material.csv', sep=",", 
	usecols=['userId','action','questionType','resourceName','resourceType', 'textMarkerChapterId','userAnswerCorrect','MinuserAnswerDuration',
    'EventMonth','EventDay','EventHour'], low_memory=False)
df = df.rename(columns={"userId": "Student ID", "action":"Action","questionType": "Question type", "resourceName":"Resource name", "resourceType":"Resource type", 
    "textMarkerChapterId": "Topic ID", "userAnswerCorrect":"Student answer", "MinuserAnswerDuration": "Answer duration", "EventMonth": "Monthly activity", 
    "EventDay": "Daily activity", "EventHour": "Hourly activity"})
df = df[np.isfinite(df['Topic ID'])]
df['Topic ID'] = df['Topic ID'].astype(int)
df['Monthly activity'] = df['Monthly activity'].astype(int)

df1 =  df[(df['Answer duration'] > 0) & (df['Answer duration'] < 20)]
df1 =  df1[(df1['Topic ID'] > 600) & (df1['Topic ID'] < 3500)]
df1['Student answer'] = df1['Student answer'].replace({True: 'Correct', False: 'Incorrect'})
df1['Answer duration'] = round(df1['Answer duration'], 2)


for i in range(10000):
    df1 = df1.sample(10000)

df1 = df1.sort_values('Monthly activity', ascending=True)
T = df1['Student ID'][df1['Student answer'] == 'Correct'].count()
F = df1['Student ID'][df1['Student answer'] == 'Incorrect'].count()


# dropdown options
slice = df1.iloc[:,[0, 5, 9, 10]]
opts = [{'label' : i, 'value' : i} for i in slice]

group_labels  = ['Answer duration']
colors = ['#3c19f0']
axis_labels =  ['1', '2', '3', '4', '5', '6', '7', '8', '9' , '10', '11', '12']

df_predict = df1.iloc[:,[0, 5, 6, 7, 8, 9, 10]]
df_predict['Student answer'] = df_predict['Student answer'].replace({'Correct': 1, 'Incorrect': 0})

df_X = df_predict
df_Y = df_predict.iloc[:,[4]]

X = df_X
Y = df_Y.values.ravel()

# shuffle data
X, y = shuffle(X, Y, random_state=0)

# ---------------- Methods for creating components in the layout code
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

# ----------------- Answer duration distribution
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
duration_fig.update_layout(title_text="Answer duration distribution (M=%.2f, SD=%.2f)" %(sample.mean(),sample.std()), height = 400)

# ------------------------------- Dashboard ------------------------------- 
app.layout = html.Div([
    # ----------------- Header
    html.Div([
		html.Div([
		    html.H1(children='An Inteactive Dashboard for Educational Datasets', className='eleven columns'),
		    html.Img(
		        src="https://upload.wikimedia.org/wikipedia/commons/a/a7/Linneuniversitetet_logo.png",
		        className='one column',
		        style={
		            'height': '7%',
		            'width': '7%',
		            'float': 'right',
		            'position': 'relative',
		            'padding-top': 0,
		            'padding-right': 0
		        },
		    ),
		    html.Div(children='''
		                A web application framework for an imbalanced dataset including the learning behaviors of 6,423 students who used an online study tool with different educational topics for the period of one year.
		                ''',
		                className='ten columns'
		        )
		    ], className="row",
		    style = {'padding' : '20px' ,
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
                    html.P("No. of student IDs"),
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
                    html.P("No. of topic IDs"),
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
                    'padding-top': '30px',
                    'padding-down':'15px',
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
                html.H6("T-SNE Parameters"),
                html.Div(
                    children=[
                    Card([
                        NamedSlider(
                            # Dimension of the embedded space
                            name="Number of Dimensions",
                            short="n-components",
                            min=2,
                            max=3,
                            step=None,
                            val=2,
                            marks={
                                i: str(i) for i in [2, 3]
                            },
                        ),
                        NamedSlider(
                            # 
                            name="Number of Iterations",
                            short="iterations",
                            min=250,
                            max=1000,
                            step=None,
                            val=750,
                            marks={
                                i: str(i) for i in [250, 500, 750, 1000]
                            },
                        ),
                        NamedSlider(
                            # number of nearest neighbors that is used in other manifold learning algorithms
                            name="Perplexity",
                            short="perplexity",
                            min=5,
                            max=100,
                            step=None,
                            val=15,
                            marks={i: str(i) for i in [5, 15, 30, 50, 100]},
                        ),
                        NamedSlider(
                            # If the learning rate is too high, the data may look like a ‘ball’ with any point approximately 
                            # equidistant from its nearest neighbours. If the learning rate is too low, most points may look 
                            # compressed in a dense cloud with few outliers.
                            name="Learning Rate",
                            short="learning-rate",
                            min=10,
                            max=600,
                            step=None,
                            val=200,
                            marks={i: str(i) for i in [10, 100, 200, 400, 600]},
                        ),
                        ])
                        ],
                    ),
                ], className= 'two columns',
                style = {
                    'padding-left': '30px',
                    'padding-right':'20px',
                    'padding-top': '20px',
                    'padding-down':'20px',
                    'height' : 400,
                    'backgroundColor' : 'white',
                    'color': '#3c4861'}
                ),
            html.Div(
                className='fivehalf columns',
                children=dcc.Graph(
                    id='graph-2d-plot-tsne'
                )
            ),

            html.Div([
                html.Div([
                    html.P("Random Forest classifier"),
                    ],
                    style = {
                        'padding-top': '20px',
                        'padding-left': '180px'}
                ),
                html.Div(
                    dcc.Graph(id='predict'),
                ),            
            ], 
            className= 'fourhalf columns',
            style = {
                'padding-left': '15px',
                'padding-right': '15px',
                'backgroundColor' : 'white',
                'color': '#3c4861'}
            ), 
        ],  
        className="row",
        style = {
            'padding-top': '15px',
            'padding-down':'15px',
            'padding-right': '20px',
            'padding-left':'20px',
            'backgroundColor' : '#d4ddee',
            'color': 'black'}
        )
    ]),

    dcc.Tabs([
        dcc.Tab(label='Student Answer', children=[
            # --------------- scatter plots, pie chart and dropdown
            html.Div([ 
                html.Div([
                    html.Div(
                        className='nine columns',
                        children=dcc.Graph(
                            id='cluster1'
                            )
                    ),
                    html.Div([
                        html.Div([
                            html.H6("Choose a feature:"),
                            dcc.Dropdown(id = 'opt1', options = opts, value = 'Student ID'), 
                            ],
                            style = {
                                'padding-top': '15px',
                                'padding-right': '40px',
                                'padding-left':'20px'}
                        ),
                        html.Div([
                            dcc.Graph(id='userAnswerCorrect-pie'),
                            ],
                            style = {
                                'padding-left': '15px',
                                'padding-right': '30px'}
                        ),            
                    ], 
                    className= 'three columns',
                    style = {
                        'padding-left': '15px',
                        'backgroundColor' : 'white',
                        'color': '#3c4861'}
                    ), 
                ],  
                className="row",
                style = {
                    'padding-down':'15px',
                    'padding-right': '20px',
                    'padding-left':'20px',
                    'backgroundColor' : '#d4ddee',
                    'color': 'black'}
                )
            ]),
        ]),

    	dcc.Tab(label='General Distribution', children=[
            # --------------- histograms and Radio boxes
            html.Div([       
                html.Div([
                	html.Div([
                        dcc.Graph(
                            id='bar1'
                        )], className= 'ten columns'
                        ),
                    html.Div([
                        html.H6("Select a feature:"),
                    	dcc.RadioItems(id = 'items', 
                    		options=[
                            {'label': 'Student ID', 'value': 'Student ID'},
                            {'label': 'Topic ID', 'value': 'Topic ID'},
                            {'label': 'Question type', 'value': 'Question type'},
                            {'label': 'Answer duration', 'value': 'Answer duration'},
                            {'label': 'Monthly activity', 'value': 'Monthly activity'},
                            {'label': 'Daily activity', 'value': 'Daily activity'},
                            {'label': 'Hourly activity', 'value': 'Hourly activity'}
                        ], 
                    		value = 'Answer duration'), 
                        ], className= 'two columns',
                        style = {
                            'padding-left': '20px',
                            'padding-right':'20px',
                            'padding-top': '60px',
                        	'padding-down':'20px',
                        	'height' : 400,
                            'backgroundColor' : 'white',
                            'color': '#3c4861'}
                        ),
                    ],  className="row",
                    style = {
                        'padding-down':'15px',
                        'padding-right': '20px',
                        'padding-left':'20px',
                        'backgroundColor' : '#d4ddee',
                        'color': 'black'}
                    )
            ])
        ]),

        dcc.Tab(label='Answer Duration', children=[
            # --------------- Scatter plots and distribution plots
            html.Div([       
                html.Div([
                    html.Div([
                        dcc.Graph(
                            id='MinuserAnswerDuration-scatter'
                        )], className= 'seven columns'
                        ),
                    html.Div([
                        dcc.Graph(
                            id='duration-distribution',
                            figure = duration_fig
                        )
                        ], className= 'five columns',
                        style = {
                            'backgroundColor' : '#d4ddee',
                            'color': 'black',
                            }
                        )
                    ],  className="row",
                    style = {
                        'padding-down':'15px',
                        'padding-right': '20px',
                        'padding-left':'20px',
                        'backgroundColor' : '#d4ddee',
                        'color': 'black'}
                    )
            ])
        ]),

        dcc.Tab(label='Topic Distribution', children=[
            # ---------------  histograms and scatter plots
            html.Div([       
                html.Div([
                    html.Div([
                        dcc.Graph(
                            id='StudentAnswer-histogram'
                        )], className= 'three columns'
                        ),
                    html.Div([
                        dcc.Graph(
                            id='CorrectAnswer-histogram'
                        )], className= 'three columns'
                        ),
                    html.Div([
                        dcc.Graph(
                            id='studentanswer-bar',
                        )], className= 'six columns'
                        )
                    ],  className="row",
                    style = {
                        'padding-down':'15px',
                        'padding-right': '20px',
                        'padding-left':'20px',
                        'backgroundColor' : '#d4ddee',
                        'color': 'black'}
                    )
            ])
        ]),

        dcc.Tab(label='Activity', children=[
            # --------------- Pie charts and box plots
            html.Div([       
                html.Div([
                    html.Div([
                        dcc.Graph(
                            id='EventMonth-pie'
                        )], className= 'three columns'
                        ),
                    html.Div([
                        dcc.Graph(
                            id='questionType-pie'
                        )], className= 'threehalf columns'
                        ),
                    html.Div([
                        dcc.Graph(
                            id='EventMonth-box'
                        )], className= 'fivehalf columns',
                        )
                    ],  className="row",
                    style = {
                        'padding-down':'15px',
                        'padding-right': '20px',
                        'padding-left':'20px',
                        'backgroundColor' : '#d4ddee',
                        'color': 'black'}
                    )
            ])
        ])   
    ],
    style = {
        'padding-top': '15px',
        'padding-left': '20px',
        'padding-right': '20px',
        'backgroundColor' : '#d4ddee',
        }
    ),

    # --------------- Footer 
    html.Div([
        html.Div([
            html.P('Produced by Zeynab (Artemis) Mohseni', className='ten columns'),
            html.P('Linnaeus University, Spring 2020', 
                className='two columns', 
                style={
                    'float': 'right',
                    'position': 'relative'
                    },
            )
        ], className="row",
        style = {
		    'padding': '20px',
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

# ------------------ sliders to TSNE
@app.callback(
    Output("graph-2d-plot-tsne", "figure"),
    [
        Input("slider-n-components", "value"),
        Input("slider-iterations", "value"),
        Input("slider-perplexity", "value"),
        Input("slider-learning-rate", "value"),
    ],
)
def display_2d_scatter_plot(n_components,iterations, perplexity, learning_rate):

    tsne = TSNE(
        n_components=n_components, 
        n_iter=iterations,
        learning_rate=learning_rate,
        perplexity=perplexity,
    )

    tsne_results = tsne.fit_transform(X) 

    fig = px.scatter(
        data_frame = df_X,
        x= tsne_results[:,0],
        y= tsne_results[:,1],
        color= df1['Monthly activity'],
        height = 400,
        title = 'T-SNE for student monthly activity',
        labels ={'x':'', 'y':''},
        color_continuous_scale='Picnic'
    )   
    return fig

# ------------------------ Heatmap
def heatmap1(selected_data, column):
    if selected_data:
        indices = [point['pointIndex'] for point in selected_data['points']]
        dff = df1.iloc[indices, :]
    else:
        dff = df1

    df_X = df_predict

    df_Y = df_predict.iloc[:,[4]]

    X = df_X
    Y = df_Y.values.ravel()

    # shuffle data
    X, y = shuffle(X, Y, random_state=0)
    number = df_X[column].value_counts().max()

    model = RandomForestClassifier(n_estimators=100)
    y_pred = cross_val_predict(model, X, y, cv=10)
    # calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    # confusion matrix
    conf_mx = confusion_matrix(y, y_pred)
    conf_mx1 = (conf_mx/number*100).round()
    fig = ff.create_annotated_heatmap(conf_mx1, colorscale='Picnic', x=axis_labels, y=axis_labels, showscale= True)
    fig.update_layout(title_text="Classification accuracy for students monthly activities = %.2f" %(accuracy*100) +"%", titlefont=dict(size=15),
        yaxis_title = "Month", 
        height = 349,
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

    dff1 = dff.sort_values('Student ID', ascending=True)
    if  input1 == 'Student ID':
        fig = px.scatter(
            data_frame = dff1,
            x="Student ID",
            color="Student answer",
            #size="Answer duration", 
            height = 400,
            size_max=20, 
            hover_data=dff1.columns,
            title = 'Students answers according to student ID',
            labels ={'Student ID':'Student ID'}
        )

    dff1 = dff.sort_values('Topic ID', ascending=True)
    if  input1 == 'Topic ID':
        fig = px.scatter(
            data_frame = dff1,
            x="Topic ID",
            color="Student answer",
            #size="Answer duration", 
            height = 400,
            size_max=20,
            log_x=True, 
            hover_data=dff1.columns,
            title = 'Students answers according to topic ID',
            labels ={'Topic ID':'Topic ID'}
        )

    dff1 = dff.sort_values('Daily activity', ascending=True)
    if  input1 == 'Daily activity':
        fig = px.scatter(
            data_frame = dff1,
            x="Daily activity",
            color="Student answer",
            #size="Answer duration", 
            height = 400,
            size_max=20,
            log_x=True, 
            hover_data=dff1.columns,
            title = 'Students answers according to student daily activity',
            labels ={'Daily activity':'Day'}
        )

    dff1 = dff.sort_values('Hourly activity', ascending=True)
    if  input1 == 'Hourly activity':
        fig = px.scatter(
            data_frame = dff1,
            x="Hourly activity",
            color="Student answer",
            #size="Answer duration", 
            height = 400,
            size_max=20,
            log_x=True, 
            hover_data=dff1.columns,
            title = 'Students answers according to student hourly activity',
            labels ={'Hourly activity':'Hour'}
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
        height = 304,
        title = 'Students answers (%)',
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

    if  input1 == 'Monthly activity':
        fig = px.histogram(
            data_frame = dff,
            x="Monthly activity",
            color="Student answer", 
            marginal="rug", 
            hover_data=dff.columns,
            labels ={'Monthly activity':'Month'},
            title = 'Distribution of studnets monthly activity',
            height = 400
        )

    if  input1 == 'Daily activity':
        fig = px.histogram(
            data_frame = dff,
            x="Daily activity",
            color="Student answer", 
            marginal="rug", 
            hover_data=dff.columns,
            labels ={'Daily activity':'Day'},
            title = 'Distribution of studnets daily activity',
            height = 400
        )

    if  input1 == 'Hourly activity':
        fig = px.histogram(
            data_frame = dff,
            x="Hourly activity",
            color="Student answer", 
            marginal="rug", 
            hover_data=dff.columns,
            labels ={'Hourly activity':'Hour'},
            title = 'Distribution of studnets hourly activity',
            height = 400
        )

    if  input1 == 'Answer duration':
        fig = px.histogram(
            data_frame = dff,
            x="Answer duration",
            color="Student answer", 
            marginal="rug", 
            hover_data=dff.columns,
            labels ={'Answer duration':'Answer duration'},
            title = 'Distribution of student answer duration',
            height = 400
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
            height = 400
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
            height = 400
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
            height = 400
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
        animation_frame="Monthly activity", 
        animation_group="Daily activity",
        size="Answer duration", 
        color="Student answer", 
        hover_name="Student ID",
        facet_col="Student answer",
        log_x=True, 
        size_max=35, 
        range_x=[0.01,20], 
        range_y=[600,3501],
        height = 400,
        title = 'Students answer duration',
        labels ={'Answer duration':'Students answer duration (min)', 'Topic ID':'Topic ID'}
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
            y= column,
            nbins = 30,
            title = 'Topic distribution for student answers',
            labels ={'Topic ID':'Topic IDs', 'Student answer':'student answers'},
            height = 400
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
            y= column,
            nbins = 30,
            title = 'Topic distribution for correct answers',
            labels ={'Topic ID':'Topic IDs', 'Student answer':'student correct answers'},
            height = 400
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
                    x=sorted(dff['Topic ID'].value_counts().head(10).index), 
                    y=dff['Topic ID'].value_counts().head(10),
                    mode='lines+markers',
                    name = 'Student answers',
                    opacity=0.8,
                    marker={
                        'size': 9})
    trace_2 = go.Scatter(
                        x = sorted(dff['Topic ID'][dff['Student answer'] == 'Correct'].value_counts().head(10).index), 
                        y = dff['Topic ID'][dff['Student answer'] == 'Correct'].value_counts().head(10), 
                        mode='lines+markers',
                        name = 'Correct answers',
                        opacity=0.8,
                        marker={
                            'size': 9})

    layout = go.Layout(title = 'Top 10 topics', 
        hovermode = 'closest', 
        height= 400, 
        xaxis_title="Topic IDs",
        yaxis_title="Count of student answers")
    fig = go.Figure(data = [trace_1, trace_2], layout = layout)
    return fig

# ------------------------pie plot-monthly activity definition - 5th tab: Activity
def pie2(selected_data, column):
    if selected_data:
        indices = [point['pointIndex'] for point in selected_data['points']]
        dff = df1.iloc[indices, :]
    else:
        dff = df1

    fig = px.pie(
        data_frame = dff,
        names = column,
        hole = .3,
        height = 400,
        title = 'Students activity per month (%)',
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
        height = 400,
        title = 'Question types (%)',
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
        height = 400,
        title= 'Distribution of students activity per month',
        labels ={'Monthly activity':'Students activity per month', 'Topic ID':'Topic IDs'},
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
    return heatmap1(selected_data, 'Monthly activity')

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
    return pie2(selected_data, 'Monthly activity')

@app.callback(
    Output('questionType-pie', 'figure'),
    [Input('graph-2d-plot-tsne', 'selectedData')])
def update_QT(selected_data):
    return pie3(selected_data, 'Question type')

@app.callback(
    Output('EventMonth-box', 'figure'),
    [Input('graph-2d-plot-tsne', 'selectedData')])
def update_EventMonth(selected_data):
    return box(selected_data, 'Monthly activity')

if __name__ == '__main__':
    app.run_server(debug=True)
