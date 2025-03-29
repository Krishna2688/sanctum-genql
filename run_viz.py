import streamlit as st
import pandas as pd
import plotly.express as px
from time import sleep

if "fig" not in st.session_state:
    st.session_state["fig"] = None

if "code" not in st.session_state:
    st.session_state["code"] = None

# Header Section with Logo
col1, col2 = st.columns([1, 16])
with col1:
    st.image("logo_t.png", width=100)  # Replace 'path_to_logo.png' with the actual path to your logo image
with col2:
    st.title(":orange[Vizualization] Viewer")

# Text area for user to input Python code
st.subheader("Enter your Python Viz code")
user_code = st.text_area("Paste your Python code here:", height=200, key="viz_code")
run_viz_button = st.button("Run Code")

# Button to execute the code
if run_viz_button:
    try:
        with st.expander("code"):
            st.code(user_code)

        # Define a custom namespace to execute the code
        exec_namespace = {}
        with st.spinner("Generating Viz.."):
            sleep(5)
            exec(user_code, {"pd": pd, "px": px, "st": st}, exec_namespace)
            with st.expander("Viz"):
                st.plotly_chart(exec_namespace["fig"])

        st.session_state["code"] = user_code
        st.session_state["fig"] = exec_namespace["fig"]
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    if st.session_state["code"] is not None:
        with st.expander("code"):
            st.code(st.session_state["code"])
    if st.session_state["fig"] is not None:
        with st.expander("viz"):
            st.plotly_chart(st.session_state["fig"])


