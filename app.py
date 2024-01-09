# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
 
st.title("Simple Streamlit App")
 
# Create a DataFrame
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100)
})
 
# Scatter Chart with Altair
chart = alt.Chart(df).mark_circle().encode(
    x='x',
    y='y',
    tooltip=['x', 'y']
).interactive()
 
# Display the Chart
st.altair_chart(chart, use_container_width=True)