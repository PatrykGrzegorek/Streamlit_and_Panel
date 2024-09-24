import streamlit as st
import pandas as pd
import stAppPlotly.express as px

def load_iris_data():
    return px.data.iris()

def load_election_data():
    return px.data.election()

st.title('Data Analysis Application')

# Section selection
section = st.sidebar.selectbox("Select Analysis Section", ["Iris Analysis", "Election Analysis"])

if section == "Iris Analysis":
    st.header('Iris Dataset Analysis')
    df = load_iris_data()

    if st.checkbox('Show raw data (Iris)'):
        st.write(df)

    columns = df.columns.tolist()
    x_axis = st.selectbox('Select X axis', columns, key='x1')
    y_axis = st.selectbox('Select Y axis', columns, key='y1')

    fig = px.scatter(df, x=x_axis, y=y_axis)
    st.plotly_chart(fig)

elif section == "Election Analysis":
    st.header('Election Dataset Analysis')
    df = load_election_data()

    if st.checkbox('Show raw data (Election)'):
        st.write(df)

    fig = px.scatter_3d(df, x="Joly", y="Coderre", z="Bergeron", color="winner", size="total", hover_name="district",
                        symbol="result", color_discrete_map={"Joly": "blue", "Bergeron": "green", "Coderre": "red"})
    st.plotly_chart(fig)
