import streamlit as st
import pandas as pd
from streamlit_extras.bottom_container import bottom

if "SQL" not in st.session_state:
    st.session_state["SQL"] = None

if "conn" not in st.session_state:
    st.session_state["conn"] = None

if "Data" not in st.session_state:
    st.session_state["Data"] = pd.DataFrame()

if "run" not in st.session_state:
    st.session_state["run"] = True

# Header Section with Logo
col1, col2 = st.columns([1, 16])
with col1:
    st.image("logo_t.png", width=100)  # Replace 'path_to_logo.png' with the actual path to your logo image
with col2:
    st.title("Try your :orange[SQL]")

def enable():
    st.session_state.run = False

if st.session_state["conn"]:

    SQL = st.text_area("Paste your SQL Query from SQL WIZ", height=200, on_change=enable())
    st.button("Run SQL Query", disabled=st.session_state.run, key="run_sql")

    if SQL:
        st.session_state.run = False
        if "run_sql" in st.session_state and st.session_state.run_sql == True:
            st.code(SQL)
            with st.spinner("executing.."):
                st.session_state.run = True
                result = pd.read_sql(SQL, st.session_state["conn"])
                st.dataframe(result.head(10), use_container_width=True, hide_index=True)
                st.session_state["SQL"] = SQL
                st.session_state["Data"] = result
    else:
        st.code(st.session_state["SQL"], language="sql")
        st.dataframe(st.session_state["Data"].head(10), use_container_width=True, hide_index=True)




else:
    st.error("**👈 To Begin Connect to DB from the sidebar** ")

with bottom():
    st.write("**&copy; 2025 Sanctum Digital Solutions**")