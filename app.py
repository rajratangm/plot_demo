import streamlit as st 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from pygwalker.api.streamlit import StreamlitRenderer
import altair as alt
import pandas as pd 
import plotly.express as px
from bokeh.plotting import figure
from bokeh.palettes import Spectral11
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.palettes import Spectral11
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datashader as ds
import datashader.transfer_functions as tf
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datashader as ds
import datashader.transfer_functions as tf
from PIL import Image

st.set_page_config(layout='wide')
@st.cache_data
def load_data():
    data = pd.read_csv('DiffCost.csv', encoding="cp932")
    st.header(f'Rows:-{data.shape[0]} and Columns:-{data.shape[1]}')
    return data

def pygwalker():
    
    # Add Title
    st.title("Use Pygwalker In Streamlit")

    start_time = time.time()
    df = st.session_state.data

    # Time for Pygwalker visualization
    start_time = time.time()
    renderer = StreamlitRenderer(df)  # Create the renderer for the dataset
    pygwalker_time = time.time() - start_time

    # Display the time taken for Pygwalker visualization
    st.write(f"Time taken for Pygwalker visualization: {pygwalker_time:.2f} seconds")

    # Show the Pygwalker interactive explorer
    renderer.explorer()


def altair():

    df = st.session_state.data

    # Line plot
    if st.session_state.plot =='linear':
        start_time = time.time()
        line_chart = alt.Chart(df).mark_line().encode(
            x='InhousePowerRate',
            y='UnplannedOutageRate',
            
        ).properties(title="Line Plot")

        st.altair_chart(line_chart)
        plot_time_line = time.time() - start_time
        st.write(f"Time taken to create Line Plot: {plot_time_line:.2f} seconds")

    # Scatter plot
    if st.session_state.plot =='scattered':
        start_time = time.time()
        scatter_chart = alt.Chart(df).mark_point().encode(
            x='InhousePowerRate',
            y='UnplannedOutageRate',
        ).properties(title="Scatter Plot")

        st.altair_chart(scatter_chart)
        plot_time_scatter = time.time() - start_time
        st.write(f"Time taken to create Scatter Plot: {plot_time_scatter:.2f} seconds")

    # Bar chart
    if st.session_state.plot =='bar':
        start_time = time.time()
        bar_chart = alt.Chart(df).mark_bar().encode(
            x='MustRun',
            y='UnplannedOutageRate'
        ).properties(title="Bar Chart")

        st.altair_chart(bar_chart)
        plot_time_bar = time.time() - start_time
        st.write(f"Time taken to create Bar Chart: {plot_time_bar:.2f} seconds")


def  plot_plotly():

    df = st.session_state.data
   

    # Line plot
    if st.session_state.plot =='linear':
        start_time = time.time()
        line_chart = px.line(df, x='InhousePowerRate', y='UnplannedOutageRate', title="Line Plot")
        st.plotly_chart(line_chart)
        plot_time_line = time.time() - start_time
        st.write(f"Time taken to create Line Plot: {plot_time_line:.2f} seconds")

    # Scatter plot
    elif st.session_state.plot=='scattered':
        start_time = time.time()
        scatter_chart = px.scatter(df, x='InhousePowerRate', y='UnplannedOutageRate', title="Scatter Plot")
        st.plotly_chart(scatter_chart)
        plot_time_scatter = time.time() - start_time
        st.write(f"Time taken to create Scatter Plot: {plot_time_scatter:.2f} seconds")

    # Bar chart
    if st.session_state.plot=='bar':
        start_time = time.time()
        bar_chart = px.bar(df, x='MustRun', y='UnplannedOutageRate', title="Bar Chart")
        st.plotly_chart(bar_chart)
        plot_time_bar = time.time() - start_time
        st.write(f"Time taken to create Bar Chart: {plot_time_bar:.2f} seconds")


def matplotlib():

    df = st.session_state.data


    # ----------- Line Plot ----------- #
    if st.session_state.plot =='linear':
        start_time = time.time()

        # Line plot using Matplotlib
        plt.figure(figsize=(10, 6))
        plt.plot(df['InhousePowerRate'], df['UnplannedOutageRate'], label='UnplannedOutageRate vs InhousePowerRate', color='blue')
        plt.title('Line Plot: UnplannedOutageRate vs InhousePowerRate')
        plt.xlabel('InhousePowerRate')
        plt.ylabel('UnplannedOutageRate')
        plt.legend()

        # Display the plot using Streamlit
        st.pyplot(plt)

        plot_time_line = time.time() - start_time
        st.write(f"Time taken to create Line Plot: {plot_time_line:.2f} seconds")

    # ----------- Bar Chart ----------- #
    if st.session_state.plot =='bar':
        start_time = time.time()

        # Bar plot using Matplotlib
        plt.figure(figsize=(10, 6))
        df_bar = df.groupby('MustRun')['UnplannedOutageRate'].mean().reset_index()  # Grouping by 'MustRun' to plot the averInhousePowerRate UnplannedOutageRate
        plt.bar(df_bar['MustRun'], df_bar['UnplannedOutageRate'], color='green')
        plt.title('Bar Chart: AverInhousePowerRate UnplannedOutageRate by MustRun')
        plt.xlabel('MustRun')
        plt.ylabel('AverInhousePowerRate UnplannedOutageRate')

        # Display the plot using Streamlit
        st.pyplot(plt)

        plot_time_bar = time.time() - start_time
        st.write(f"Time taken to create Bar Chart: {plot_time_bar:.2f} seconds")

    # ----------- Scatter Plot ----------- #
    if st.session_state.plot =='scattered':
        start_time = time.time()

        # Scatter plot using Matplotlib
        plt.figure(figsize=(10, 6))
        plt.scatter(df['InhousePowerRate'], df['UnplannedOutageRate'], label='UnplannedOutageRate vs InhousePowerRate', color='red')
        plt.title('Scatter Plot: UnplannedOutageRate vs InhousePowerRate')
        plt.xlabel('InhousePowerRate')
        plt.ylabel('UnplannedOutageRate')
        plt.legend()

        # Display the plot using Streamlit
        st.pyplot(plt)

        plot_time_scatter = time.time() - start_time
        st.write(f"Time taken to create Scatter Plot: {plot_time_scatter:.2f} seconds")



# def plot_bokeh():
#     # Caching the dataset loading to avoid reloading it every time


#     # Load dataset and measure the time
#     df = st.session_state.data

#     # ----------- Line Plot ----------- #
#     if st.session_state.plot=='linear':
#         start_time = time.time()

#         # Line plot using Bokeh
#         p_line = figure(title="Line Plot: UnplannedOutageRate vs InhousePowerRate", x_axis_label='InhousePowerRate', y_axis_label='UnplannedOutageRate', height=400, width=700)
#         p_line.line(df['InhousePowerRate'], df['UnplannedOutInhousePowerRateRate'], legend_label="UnplannedOutageRate", line_width=2, color=Spectral11[0])

#         # Display the line chart using Streamlit
#         st.bokeh_chart(p_line)

#         plot_time_line = time.time() - start_time
#         st.write(f"Time taken to create Line Plot: {plot_time_line:.2f} seconds")

#     # ----------- Bar Chart ----------- #
#     if st.session_state.plot =='bar':
#         start_time = time.time()

#         # Group by 'MustRun' and calculate averInhousePowerRate UnplannedOutageRate
#         df_bar = df.groupby('MustRun')['UnplannedOutageRate'].mean().reset_index()

#         # Convert 'MustRun' to categorical type for Bokeh
#         df_bar['MustRun'] = df_bar['MustRun'].astype(str)  # Ensure it's a string type for categoricals

#         # Bar plot using Bokeh
#         p_bar = figure(title="Bar Chart: AverInhousePowerRate UnplannedOutageRate by MustRun", x_axis_label='MustRun', y_axis_label='AverInhousePowerRate UnplannedOutageRate', height=400, width=700)
#         p_bar.vbar(x=df_bar['MustRun'], top=df_bar['UnplannedOutageRate'], width=0.4,  legend_label="AverInhousePowerRate UnplannedOutageRate")

#         # Display the bar chart using Streamlit
#         st.bokeh_chart(p_bar)

#         plot_time_bar = time.time() - start_time
#         st.write(f"Time taken to create Bar Chart: {plot_time_bar:.2f} seconds")

#     # ----------- Scatter Plot ----------- #
#     if st.session_state.plot =='scattered':
#         start_time = time.time()

#         # Scatter plot using Bokeh
#         p_scatter = figure(title="Scatter Plot: UnplannedOutageRate vs InhousePowerRate", x_axis_label='InhousePowerRate', y_axis_label='UnplannedOutageRate', height=400, width=700)
#         p_scatter.scatter(df['InhousePowerRate'], df['UnplannedOutageRate'], size=8, color=Spectral11[2], legend_label="UnplannedOutageRate")

#         # Display the scatter plot using Streamlit
#         st.bokeh_chart(p_scatter)

#         plot_time_scatter = time.time() - start_time
#         st.write(f"Time taken to create Scatter Plot: {plot_time_scatter:.2f} seconds")


def plot_bokeh_():
    # Caching the dataset loading to avoid reloading it every time

    # Load dataset and measure the time
    df = st.session_state.data

    # ----------- Line Plot ----------- #
    if st.session_state.plot =='linear':
        start_time = time.time()

        # Line plot using Bokeh
        p_line = figure(title="Line Plot: UnplannedOutageRate vs InhousePowerRate", x_axis_label='InhousePowerRate', y_axis_label='UnplannedOutageRate', height=400, width=700)
        p_line.line(df['InhousePowerRate'], df['UnplannedOutageRate'], legend_label="UnplannedOutageRate", line_width=2, color=Spectral11[0])

        # Display the line chart using Streamlit
        st.bokeh_chart(p_line)

        plot_time_line = time.time() - start_time
        st.write(f"Time taken to create Line Plot: {plot_time_line:.2f} seconds")

    # ----------- Bar Chart ----------- #
    if st.session_state.plot =='bar':
        start_time = time.time()

        # Group by 'MustRun' and calculate averInhousePowerRate UnplannedOutageRate
        df_bar = df.groupby('MustRun')['UnplannedOutageRate'].mean().reset_index()

        # Bar plot using Bokeh
        p_bar = figure(title="Bar Chart: AverInhousePowerRate UnplannedOutageRate by MustRun", x_axis_label='MustRun', y_axis_label='AverInhousePowerRate UnplannedOutageRate', height=400, width=700)
        p_bar.vbar(x=df_bar['MustRun'], top=df_bar['UnplannedOutageRate'], width=0.4, color=Spectral11[1], legend_label="AverInhousePowerRate UnplannedOutageRate")

        # Display the bar chart using Streamlit
        st.bokeh_chart(p_bar)

        plot_time_bar = time.time() - start_time
        st.write(f"Time taken to create Bar Chart: {plot_time_bar:.2f} seconds")

    # ----------- Scatter Plot ----------- #
    if st.session_state.plot =='scattered':
        start_time = time.time()

        # Scatter plot using Bokeh
        p_scatter = figure(title="Scatter Plot: UnplannedOutageRate vs InhousePowerRate", x_axis_label='InhousePowerRate', y_axis_label='UnplannedOutageRate', height=400, width=700)
        p_scatter.scatter(df['InhousePowerRate'], df['UnplannedOutageRate'], size=8, color=Spectral11[2], legend_label="UnplannedOutageRate")

        # Display the scatter plot using Streamlit
        st.bokeh_chart(p_scatter)

        plot_time_scatter = time.time() - start_time
        st.write(f"Time taken to create Scatter Plot: {plot_time_scatter:.2f} seconds")



# library = st.selectbox('select plotting lybrary', ['plotly','pygwalker','bokeh','matplotlib','altair'])
# plot = st.selectbox('select plot', ['linear', 'scattered','bar'])
# st.session_state.data = load_data()
# if plot == 'linear':
#     st.session_state.plot = 'linear'
# if plot =='scattered':
#     st.session_state.plot ='scattered'
# if plot =='bar':
#     st.session_state.plot ='bar'

# if library=='plotly':
#     plot_plotly()
# elif library=='pygwalker':
#     pygwalker()
# elif library=='bokeh':
#     plot_bokeh_()
# elif library=='matplotlib':
#     matplotlib()
# elif library=='altair':
#     altair()

    
def datashader_plot():
    # Load dataset
    df = st.session_state.data

    # Aggregation function for datashader scatter plot
    def create_image(x_range, y_range, w=700, h=500):
        cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
        agg = cvs.points(df, 'InhousePowerRate', 'UnplannedOutageRate')
        img = tf.shade(agg, cmap=["lightblue", "darkblue"], how="linear").to_pil()
        return img

    if st.session_state.plot == 'scattered':
        start_time = time.time()

        # Define range of the data
        x_range = (df['InhousePowerRate'].min(), df['InhousePowerRate'].max())
        y_range = (df['UnplannedOutageRate'].min(), df['UnplannedOutageRate'].max())

        # Generate the image using Datashader
        img = create_image(x_range, y_range)

        # Display scatter plot in Streamlit
        st.image(img, caption="Scatter Plot with Datashader", use_container_width=True)

        plot_time_scatter = time.time() - start_time
        st.write(f"Time taken to create Scatter Plot: {plot_time_scatter:.2f} seconds")

    else:
        st.write("Datashader currently supports only Scatter Plot for this implementation.")


# Add Datashader to the options
library = st.selectbox('select plotting library', ['plotly', 'pygwalker', 'bokeh', 'matplotlib', 'altair', 'datashader'])
plot = st.selectbox('select plot', ['linear', 'scattered', 'bar'])
st.session_state.data = load_data()

# Set session state for selected plot
if plot == 'linear':
    st.session_state.plot = 'linear'
elif plot == 'scattered':
    st.session_state.plot = 'scattered'
elif plot == 'bar':
    st.session_state.plot = 'bar'

# Route to the appropriate plotting function
if library == 'plotly':
    plot_plotly()
elif library == 'pygwalker':
    pygwalker()
elif library == 'bokeh':
    plot_bokeh_()
elif library == 'matplotlib':
    matplotlib()
elif library == 'altair':
    altair()
elif library == 'datashader':
    datashader_plot()
