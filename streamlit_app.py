import streamlit as st

st.set_page_config(
    page_title="GENQLWIZ",
    page_icon="sanctum_t_l.png",
    layout="wide"
)
st.logo("sanctum_t_l.png", size="large")

db_connect = st.Page("1_Connect_DB.py", title="Connect to DB", icon=":material/database:")
generate_sql = st.Page("2_SQL_WIZ.py", title="Generate SQL", icon=":material/join_inner:")
run_sql = st.Page("3_Try_SQL.py", title="Run SQL", icon=":material/terminal:")
generate_viz_code = st.Page("Viz_Wiz_Code.py", title="Generate Viz Code", icon=":material/terminal:")
generate_viz = st.Page("Viz_Wiz.py", title="Generate Viz", icon=":material/terminal:")
run_viz = st.Page("run_viz.py", title="Run Viz Code", icon=":material/scatter_plot:")

home = st.Page("landing.py", title="Home", icon=":material/home:")


pg = st.navigation(
        {
            "Home": [home],
            "DB": [db_connect],
            "SQL WiZard": [generate_sql, run_sql],
            "Viz WiZard": [generate_viz, run_viz]

        }
    )

pg.run()