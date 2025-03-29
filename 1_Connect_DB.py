import streamlit as st
from sqlalchemy import create_engine
import pandas as pd
from streamlit_extras.bottom_container import bottom

if "conn" not in st.session_state:
    st.session_state["conn"] = None

if "database" not in st.session_state:
    st.session_state["database"] = None

def connect_to_db(username, password, host, database):
    try:
        engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}/{database}")
        conn = engine.connect()
        return conn, engine
    except Exception as e:
        # st.error(f"Connection failed: {e}")
        return None, None

# st.set_page_config(page_title="Connect MySQL DB", page_icon="📈")

# UI for DB connection
st.header("Database Connection")
username = st.text_input("Username")
password = st.text_input("Password", type="password")
host = st.text_input("Host", value="localhost")
database = st.text_input("Database")

if st.button("Connect"):
    conn, engine = connect_to_db(username, password, host, database)
    st.session_state["conn"] = conn
    st.session_state["engine"] = engine
    if conn:
        st.session_state["database"] = database
        # st.success(f"Connected to the database: {st.session_state['database']}.")
    else:
        st.error(f"Failed to connect: {st.session_state['database']}.")
        st.session_state.schema = {}

# Query to get the list of tables
if st.session_state["conn"]:
    st.success(f"Connected to the database: {st.session_state['database']}.")
    try:
        query = "SHOW TABLES"
        tables = pd.read_sql(query, st.session_state["conn"])
        table_list = tables.values.flatten().tolist()
        tables.columns = ["Table Name"]  # Rename column for better readability

        st.header("List of Tables in the Database")
        st.dataframe(tables)  # Display the tables

        st.session_state.schema = {table: pd.read_sql(f"DESCRIBE {table}", st.session_state["conn"]) for table in table_list}

        # Dropdown to select a table
        selected_table = st.selectbox("Select a table to view its schema:", tables["Table Name"])

        if selected_table:
            # Query to get the table schema
            # schema_query = f"DESCRIBE {selected_table}"
            # schema = pd.read_sql(schema_query, st.session_state["conn"])

            st.subheader(f"Schema for `{selected_table}`")
            st.dataframe(st.session_state["schema"][selected_table])  # Display the schema in a nicely formatted table
    except Exception as e:
        st.error(f"Failed to retrieve tables or schema: {e}")
else:
    st.warning(f"No Database Connected")

with bottom():
    st.write("**&copy; 2025 Sanctum Digital Solutions**")