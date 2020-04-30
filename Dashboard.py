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

df1 =  df[(df['Answer duration'] > 0) & (df['Answer duration'] < 20)]
df1 =  df1[(df1['Topic ID'] > 600) & (df1['Topic ID'] < 3500)]
df1['Student answer'] = df1['Student answer'].replace({True: 'Correct', False: 'Incorrect'})
df1['Answer duration'] = round(df1['Answer duration'], 2)


for i in range(10000):
    df1 = df1.sample(10000)

T = df1['Student ID'][df1['Student answer'] == 'Correct'].count()
F = df1['Student ID'][df1['Student answer'] == 'Incorrect'].count()

df2 = df1.sort_values('Monthly activity', ascending=True)

df3 = df2.sort_values('Monthly activity', ascending=True)
df3['Monthly activity'] = df3['Monthly activity'].replace({1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 
                                               7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'})
df4 = df1[(df1['Student answer'] == 'Correct')]

# dropdown options
slice = df1.iloc[:,[0, 5, 8, 9, 10]]
opts = [{'label' : i, 'value' : i} for i in slice]

labels = ['Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5', 'Topic 6', 'Topic 7', 'Topic 8', 'Topic 9', 'Topic 10']

group_labels  = ['Answer duration']
group_labels1  = ['Topic population']
colors = ['#3c19f0']


app.layout = html.Div([

    # ----------------- Header
    html.Div([
		html.Div([
		    html.H1(children='Hypocampus Dashboard', className='eleven columns'),
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

    # ----------------- Cluster and dropdown
    html.Div([ 
        html.Div([
        	html.Div([
                html.Div([
                    html.H6("Choose a feature:"),
                    dcc.Dropdown(id = 'opt1', options = opts, value = 'Student ID'), 
                    ],
                    style = {
                        'padding-top': '15px',
					    'padding-right': '20px',
					    'padding-left':'20px'}
                ),
                html.Div([
                    dcc.Graph(id='userAnswerCorrect-pie'),
                    ],
                    style = {
					    'padding-left':'5px'}
                ),            
            ], 
            className= 'three columns',
            style = {
	            'padding-right': '15px',
	            'backgroundColor' : 'white',
	            'color': '#3c4861'}
            ),
            html.Div(
            className='nine columns',
            children=dcc.Graph(
                id='cluster1'
                )
            )
            
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
    	dcc.Tab(label='General Distribution', children=[
            # --------------- main histogram and Radio boxes
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
            # --------------- distribution and scatter plots
            html.Div([       
                html.Div([
                    html.Div([
                        dcc.Graph(
                            id='MinuserAnswerDuration-scatter'
                        )], className= 'seven columns'
                        ),
                    html.Div([
                        dcc.Graph(
                            id='duration-distribution'
                        )
                        ], className= 'five columns',
                        style = {
                            'backgroundColor' : '#d4ddee',
                            'color': 'black'}
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
            # ---------------  histograms and distribution plots
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
                            id='studentanswer-bar'
                        )], className= 'three columns'
                        ),
                    html.Div([
                        dcc.Graph(
                            id='correctAnswer-bar'
                        )], className= 'three columns',
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
            # --------------- box and pie plots
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
            html.P('Produced by Artemis Mohseni', className='ten columns'),
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


# ------------------------pie plot-user answer definition
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


# ------------------------histogram_ duration distribution definition
def histogram3(selected_data, column):
    if selected_data:
        indices = [point['pointIndex'] for point in selected_data['points']]
        dff = df1.iloc[indices, :]
    else:
        dff = df1

    sample_means = []
    population = dff[column]
    for i in range(100):
        sample = population.sample(100)
        stat1, p1 = ttest_1samp(sample, 1)
        sample_means.append(sample.mean())
    
    fig = ff.create_distplot(
        [np.array(sample_means)], 
        group_labels, 
        bin_size=.1, 
        colors=colors
    )
    fig.update_layout(title_text="Answer duration distribution (M=%.2f, SD=%.2f)" %(sample.mean(),sample.std()), height = 400)
    return fig

# ------------------------Scatter plots with slider definition
def scatter1(selected_data, column):
    if selected_data:
        indices = [point['pointIndex'] for point in selected_data['points']]
        dff = df2.iloc[indices, :]
    else:
        dff = df2
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

# ------------------------bar plot top 10 definition
def bar2(selected_data, column):
    if selected_data:
        indices = [point['pointIndex'] for point in selected_data['points']]
        dff = df1.iloc[indices, :]
    else:
        dff = df1
    fig = px.bar(
        data_frame = dff, 
        x = labels, 
        y= dff['Topic ID'].value_counts().head(10), 
        labels ={'x':'', 'y':'Student answers'},
        title = 'Students answers in top 10 topics',
        height = 400,
    )
    return fig

# ------------------------bar plot top 10 correct definition
def bar3(selected_data, column):
    if selected_data:
        indices = [point['pointIndex'] for point in selected_data['points']]
        dff = df1.iloc[indices, :]
    else:
        dff = df1

    fig = px.bar(
        data_frame = dff, 
        x = labels,  
        y= dff['Topic ID'][dff['Student answer'] == 'Correct'].value_counts().head(10), 
        labels ={'x':'', 'y':'Student correct answers'},
        title = 'Correct answers in top 10 topics',
        height = 400,
    )
    return fig

# ------------------------histogram_ user definition
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

# ------------------------histogram_ correct answer definition
def histogram2(selected_data, column):
    if selected_data:
        indices = [point['pointIndex'] for point in selected_data['points']]
        dff = df4.iloc[indices, :]
    else:
        dff = df4

    fig = px.histogram(
            data_frame = dff,
            x= "Topic ID",
            y= column,
            nbins = 30,
            title = 'Topic distribution for correct answers',
            labels ={'Topic ID':'Topic IDs', 'Student answer':'student correct answers'},
            height = 400
        ) 
    return fig

# ------------------------box plot_Monthly acitvity definition
def box(selected_data, column):
    if selected_data:
        indices = [point['pointIndex'] for point in selected_data['points']]
        dff = df3.iloc[indices, :]
    else:
        dff = df3
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

# ------------------------pie plot-monthly activity definition
def pie2(selected_data, column):
    if selected_data:
        indices = [point['pointIndex'] for point in selected_data['points']]
        dff = df3.iloc[indices, :]
    else:
        dff = df3
    fig = px.pie(
        data_frame = dff,
        names = column,
        hole = .3,
        height = 400,
        title = 'Students activity per month (%)',
    )
    return fig

# ------------------------pie questionType definition
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

# ------------------ connecting dropdown to Bar plot
@app.callback(Output('bar1', 'figure'),
             [Input('items', 'value'), Input('cluster1', 'selectedData')])

def update_image_src(input1, input2):
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

# ------------------ connecting dropdown to scatter plot
@app.callback(Output('cluster1', 'figure'),
             [Input('opt1', 'value')])

def update_image_src1(input1):

    dff = df1.sort_values('Monthly activity', ascending=True)
    if  input1 == 'Monthly activity':
        fig = px.scatter(
            data_frame = dff,
            x="Monthly activity",
            color="Student answer",
            size="Answer duration", 
            #animation_frame="Monthly activity", 
            height = 400,
            size_max=20,
            log_x=True, 
            hover_data=dff.columns,
            #hover_name="Monthly activity",
            title = 'Clustering students answers according to student monthly activity',
            labels ={'Monthly activity':'Month'}
        )

    dff = df1.sort_values('Daily activity', ascending=True)
    if  input1 == 'Daily activity':
        fig = px.scatter(
            data_frame = dff,
            x="Daily activity",
            color="Student answer",
            size="Answer duration", 
            #animation_frame="Monthly activity", 
            height = 400,
            size_max=20,
            log_x=True, 
            hover_data=dff.columns,
            #hover_name="Daily activity",
            title = 'Clustering students answers according to student daily activity',
            labels ={'Daily activity':'Day'}
        )

    dff = df1.sort_values('Hourly activity', ascending=True)
    if  input1 == 'Hourly activity':
        fig = px.scatter(
            data_frame = dff,
            x="Hourly activity",
            color="Student answer",
            size="Answer duration", 
            #animation_frame="Monthly activity", 
            height = 400,
            size_max=20,
            log_x=True, 
            hover_data=dff.columns,
            #hover_name="Hourly activity",
            title = 'Clustering students answers according to student hourly activity',
            labels ={'Hourly activity':'Hour'}
        )

    dff = df1.sort_values('Topic ID', ascending=True)
    if  input1 == 'Topic ID':
        fig = px.scatter(
            data_frame = dff,
            x="Topic ID",
            color="Student answer",
            size="Answer duration", 
            #animation_frame="Monthly activity", 
            height = 400,
            size_max=20,
            log_x=True, 
            hover_data=dff.columns,
            #hover_name="Topic ID",
            title = 'Clustering students answers according to topic ID',
            labels ={'Topic ID':'Topic ID'}
        )

    dff = df1.sort_values('Student ID', ascending=True)
    if  input1 == 'Student ID':
        fig = px.scatter(
            data_frame = dff,
            x="Student ID",
            color="Student answer",
            size="Answer duration", 
            #animation_frame="Monthly activity", 
            height = 400,
            size_max=20,
            #log_x=True, 
            hover_data=dff.columns,
            #hover_name="Student answer",
            title = 'Clustering students answers according to student ID',
            labels ={'Student ID':'Student ID'}
        )
    return fig

# ------------------ Bar plot to LEDDisplays
@app.callback(
    Output('operator-Student', 'value'),
    [Input('cluster1', 'selectedData')])
def update_N_Student(selected_data):
    return LEDDisplay1(selected_data, 'Student ID')

@app.callback(
    Output('operator-Topic', 'value'),
    [Input('cluster1', 'selectedData')])
def update_N_Topic(selected_data):
    return LEDDisplay1(selected_data, 'Topic ID')

@app.callback(
    Output('operator-CoAnswer', 'value'),
    [Input('cluster1', 'selectedData')])
def update_N_CoAnswer(selected_data):
    return LEDDisplay2(selected_data, 'Student answer')

@app.callback(
    Output('operator-IncAnswer', 'value'),
    [Input('cluster1', 'selectedData')])
def update_N_IncAnswer(selected_data):
    return LEDDisplay3(selected_data, 'Student answer')

@app.callback(
    Output('operator-min_Duration', 'value'),
    [Input('cluster1', 'selectedData')])
def update_N_min(selected_data):
    return LEDDisplay4(selected_data, 'Answer duration')

@app.callback(
    Output('operator-max_Duration', 'value'),
    [Input('cluster1', 'selectedData')])
def update_N_max(selected_data):
    return LEDDisplay5(selected_data, 'Answer duration')


# ------------------ Bar plot to first pie chart
@app.callback(
    Output('userAnswerCorrect-pie', 'figure'),
    [Input('cluster1', 'selectedData')])
def update_userAnswerCorrect(selected_data):
    return pie1(selected_data, 'Student answer')

@app.callback(
    Output('questionType-pie', 'figure'),
    [Input('cluster1', 'selectedData')])
def update_QT(selected_data):
    return pie3(selected_data, 'Question type')


# ------------------ Bar plot to histograms distribution
@app.callback(
    Output('duration-distribution', 'figure'),
    [Input('cluster1', 'selectedData')])
def update_distribution_duration(selected_data):
    return histogram3(selected_data, 'Answer duration')

# ------------------ Bar plot to Scatter plot
@app.callback(
    Output('MinuserAnswerDuration-scatter', 'figure'),
    [Input('cluster1', 'selectedData')])
def update_MinuserAnswerDuration(selected_data):
    return scatter1(selected_data, 'Answer duration')

# ------------------ Bar plot to bar plots
@app.callback(
    Output('studentanswer-bar', 'figure'),
    [Input('cluster1', 'selectedData')])
def update_top10_StudentAnswer(selected_data):
    return bar2(selected_data, 'Topic ID')

@app.callback(
    Output('correctAnswer-bar', 'figure'),
    [Input('cluster1', 'selectedData')])
def update_top10_CorrectAnswer(selected_data):
    return bar3(selected_data, 'Topic ID')

# ------------------ Bar plot to histograms 
@app.callback(
    Output('StudentAnswer-histogram', 'figure'),
    [Input('cluster1', 'selectedData')])
def update_Topic_StudentAnswer(selected_data):
    return histogram1(selected_data, 'Student answer')

@app.callback(
    Output('CorrectAnswer-histogram', 'figure'),
    [Input('cluster1', 'selectedData')])
def update_Topic_CorrectAnswer(selected_data):
    return histogram2(selected_data, 'Student answer')

# ------------------ Bar plot to box plot
@app.callback(
    Output('EventMonth-box', 'figure'),
    [Input('cluster1', 'selectedData')])
def update_EventMonth(selected_data):
    return box(selected_data, 'Monthly activity')

# ------------------ Bar plot to pie plots
@app.callback(
    Output('EventMonth-pie', 'figure'),
    [Input('cluster1', 'selectedData')])
def update_EventMonth_pie(selected_data):
    return pie2(selected_data, 'Monthly activity')

if __name__ == '__main__':
    app.run_server(debug=True)
