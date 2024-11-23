from dash import Dash, html, dcc, Output, Input
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import os

pm25_data = pd.read_csv('NYC EH Data Portal - Fine particles (PM 2.5) (full table) (1).csv')
asthma_data = pd.read_csv('NYC EH Data Portal - Adults with asthma (full table).csv')
common_geoids = set(asthma_data['GeoID']).intersection(set(pm25_data['GeoID']))
combined_data = pd.merge(asthma_data, pm25_data, on='GeoID', how='inner')
asthma_data['Age-adjusted percent'] = asthma_data['Age-adjusted percent'].str.extract(r'([\d.]+)').astype(float)
asthma_data['Number'] = asthma_data['Number'].str.replace(',', '').str.replace(r'\*', '', regex=True).astype(float)
asthma_data['Percent'] = asthma_data['Percent'].str.extract(r'([\d.]+)').astype(float)

pm25_data['TimePeriod'] = pm25_data['TimePeriod'].str.extract(r'(\d{4})').astype(int)

combined_data['10th percentile mcg/m3'].fillna(combined_data['10th percentile mcg/m3'].mean(), inplace=True)
combined_data['90th percentile mcg/m3'].fillna(combined_data['90th percentile mcg/m3'].mean(), inplace=True)

# Check if 'Age-adjusted percent' is a string column
if combined_data['Age-adjusted percent'].dtype == 'object':
    combined_data['Age-adjusted percent'] = combined_data['Age-adjusted percent'].str.extract(r'([\d.]+)').astype(float)

# Check if 'Number' is a string column
if combined_data['Number'].dtype == 'object':
    combined_data['Number'] = combined_data['Number'].str.replace(',', '').str.replace(r'\*', '', regex=True).astype(float)

# Check if 'Percent' is a string column
if combined_data['Percent'].dtype == 'object':
    combined_data['Percent'] = combined_data['Percent'].str.extract(r'([\d.]+)').astype(float)

combined_data['Log_Number'] = np.log1p(combined_data['Number'])

df = combined_data

#-------------------
# Correlation heatmap
def generate_static_barplot_image(data):
    correlation_matrix = data[['Number', 'Mean mcg/m3', 'Age-adjusted percent', 'Percent']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix for Key Variables")
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    return f"data:image/png;base64,{encoded_image}"

def generate_second_barplot_image(data):
    plt.figure(figsize=(10, 6)) 
    sns.barplot(
        data=data,
        x='Geography_x',
        y='Age-adjusted percent',
        errorbar=None,
        palette="muted",
        width=0.6
    )
    plt.title("Age-adjusted Percent for High-Case Regions")
    plt.xlabel("Geography")
    plt.ylabel("Age-adjusted Percent")
    plt.xticks(rotation=45, ha='right') 
    plt.tight_layout()  
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()

    return f"data:image/png;base64,{encoded_image}"


second_barplot_src = generate_second_barplot_image(combined_data)

static_barplot_src = generate_static_barplot_image(combined_data)

# Initialize the app
app = Dash()

# Layout with Tabs
app.layout = html.Div(
    [
        html.H1("MIS 5480 Group 3: NYU Health & Air Quality", style={'textAlign': 'center'}),
        html.H3(
            children='Team 3: Austin P Abrams, Terry Hefel, Yun-Hsien Kuo, Marcie Cleland, Wanying Li',
            style={'textAlign': 'center'}
        ),
        dcc.Tabs(
            [
                # Tab 1: Interactive Plots
                dcc.Tab(
                    label='Health & Air Quality Infographics',
                children=[
                    html.H1(
                        children='Health & Air Quality Visual Analysis by Geography',
                        style={'textAlign': 'center'}
                    ),
                    dcc.Dropdown(
                        id='dropdown-selection',
                        options=[{'label': 'Select All', 'value': 'ALL'}] +
                                [{'label': geo, 'value': geo} for geo in df['Geography_x'].unique()],
                        value=['Bronx','Brooklyn', 'Manhattan', 'Queens', 'New York City'],  # Default selection
                        multi=True,
                        style={'marginBottom': '20px', 'width': '50%'}
                    ),
                    html.Div(
                        children=[
                            dcc.Graph(
                                id='barplot',
                                style={
                                    'marginTop': '20px',
                                    'border': '2px solid black',
                                    'padding': '10px',
                                    'borderRadius': '8px',
                                    'flex': '1'
                                }
                            ),
                            dcc.Graph(
                                id='scatterplot',
                                style={
                                    'marginTop': '20px',
                                    'border': '2px solid black',
                                    'padding': '10px',
                                    'borderRadius': '8px',
                                    'flex': '1'
                                }
                            )
                        ],
                        style={
                            'display': 'flex',
                            'flexDirection': 'row',
                            'gap': '20px',
                            'justifyContent': 'space-between',
                            'alignItems': 'flex-start'
                        }
                    ),
                    html.Div(
                        children=[
                            dcc.Graph(
                                id='scatterplot-mean-vs-number',
                                style={
                                    'marginTop': '20px',
                                    'border': '2px solid black',
                                    'padding': '10px',
                                    'borderRadius': '8px'
                                }
                            )
                        ],
                        style={'marginTop': '20px'} 
                    )
                ]
                ),
                # Tab 2: Static Plots
                dcc.Tab(
                    label='Findings',
                    children=[
                        # Row for the paragraph alone
                        html.Div(
                            children=[
                                html.P(
                                    "This report examines the relationship between air quality (PM 2.5 levels) and asthma prevalence across various regions of New York City. Through visual analysis of the data, we explore whether areas with higher levels of air pollution also experience higher asthma rates. While the data shows that asthma cases vary across different geographies, we found no strong evidence that poor air quality directly influences asthma outcomes.",
                                    style={
                                        'textAlign': 'left',
                                        'fontSize': '20px',
                                        'fontWeight': 'normal',
                                        'lineHeight': '1.4',
                                        'textIndent': '20px',  
                                        'marginBottom': '0',
                                        'marginTop': '15', 
                                        'padding': '0',  
                                    }
                                ),
                                html.P(
                                    "However, this lack of a clear correlation may be due to several confounding factors that were not accounted for in the analysis. Key variables such as socioeconomic disparities, access to healthcare, and other environmental factors (e.g., housing quality, urban heat islands) could play a more significant role in determining health outcomes. These factors were not controlled for in our visual analysis, and their influence may explain the absence of a direct link between air quality and asthma rates.",
                                    style={
                                        'textAlign': 'left',
                                        'fontSize': '20px',
                                        'fontWeight': 'normal',
                                        'lineHeight': '1.4',
                                        'textIndent': '20px', 
                                        'marginBottom': '0', 
                                        'marginTop': '0',  
                                        'padding': '0', 
                                    }
                                ),
                                html.P(
                                    "The findings suggest that while air pollution may be an important factor, it is likely not the sole determinant of asthma prevalence. Further research that incorporates a broader set of variables, including socioeconomic and healthcare access data, is needed to better understand the complex relationship between environmental factors and public health outcomes.",
                                    style={
                                        'textAlign': 'left',
                                        'fontSize': '20px',
                                        'fontWeight': 'normal',
                                        'lineHeight': '1.4', 
                                        'textIndent': '20px',  
                                        'marginBottom': '10px',
                                        'marginTop': '0',
                                        'padding': '0', 
                                    }
                                ),
                            ],
                            style={
                                'display': 'flex',
                                'flexDirection': 'column',
                                'gap': '10px', 
                                'justifyContent': 'center',
                                'alignItems': 'flex-start', 
                            }
                        ),
                        # Row for the heatmap and bar chart
                        html.Div(
                            children=[
                                # Heatmap image
                                html.Div(
                                    children=[
                                        html.Img(
                                            src=static_barplot_src,
                                            style={
                                                'width': 'auto', 
                                                'height': 'auto',
                                                'display': 'block',
                                                'maxWidth': '80%', 
                                                'margin': '0 auto'
                                            }
                                        ),
                                    ],
                                    style={
                                        'display': 'flex',
                                        'justifyContent': 'center',
                                        'alignItems': 'center',
                                        'marginBottom': '20px' 
                                    }
                                ),
                                # Bar chart image
                                html.Div(
                                    children=[
                                        html.Img(
                                            src=second_barplot_src,
                                            style={
                                                'width': 'auto',
                                                'height': 'auto',  
                                                'display': 'block',
                                                'margin': '0 auto' 
                                            }
                                        ),
                                    ],
                                    style={
                                        'display': 'flex',
                                        'justifyContent': 'center', 
                                        'alignItems': 'center',
                                    }
                                ),
                            ],
                            style={
                                'display': 'flex',
                                'flexDirection': 'row',
                                'justifyContent': 'space-around',
                                'alignItems': 'center',
                            }
                        ),
                    ]
                )
            ]
        )
    ]
)

# Callbacks for Interactive Plots
@app.callback(
    [
        Output('barplot', 'figure'),
        Output('scatterplot', 'figure'),
        Output('scatterplot-mean-vs-number', 'figure')
    ],
    Input('dropdown-selection', 'value') 
)
def update_graphs(selected_geographies):
    if not selected_geographies or 'ALL' in selected_geographies:
        filtered_data = df  
    else:
        filtered_data = df[df['Geography_x'].isin(selected_geographies)]

    barplot_fig = px.bar(
        filtered_data,
        x='Geography_x',
        y='Number',
        title='Number of Cases by Geography',
        labels={'Number': 'Number of Cases', 'Geography_x': 'Geography'}
    )
    barplot_fig.update_layout(xaxis_tickangle=45)

    scatterplot_fig = px.scatter(
        filtered_data,
        x='Geography_x',
        y='Mean mcg/m3',
        color='Geography_x',
        title="Air Quality (Mean mcg/m³) by Geography",
        labels={'Geography_x': 'Geography', 'Mean mcg/m3': 'Mean mcg/m³'},
        hover_data=['Geography_x', 'Mean mcg/m3']
    )
    scatterplot_fig.update_layout(
        xaxis_title="Geography",
        yaxis_title="Mean mcg/m³",
        legend_title="Geography"
    )

    scatterplot_mean_vs_number = px.scatter(
        filtered_data,
        x='Mean mcg/m3',
        y='Number',
        color='Geography_x',
        title="Air Quality (Mean mcg/m³) vs. Number of Cases",
        labels={'Mean mcg/m3': 'Mean mcg/m³', 'Number': 'Number of Cases'}
    )
    scatterplot_mean_vs_number.update_layout(
        xaxis_title="Mean mcg/m³",
        yaxis_title="Number of Cases",
        legend_title="Geography"
    )

    return barplot_fig, scatterplot_fig, scatterplot_mean_vs_number

server = app.server

if __name__ == '__main__':
    app.run_server(debug=True)
