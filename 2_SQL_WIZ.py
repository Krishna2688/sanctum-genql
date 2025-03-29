import streamlit as st
import requests
import time
import os
import json
import csv
import io
from streamlit_extras.bottom_container import bottom

if "conn" not in st.session_state:
    st.session_state["conn"] = None

if "database" not in st.session_state:
    st.session_state["database"] = None

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "model" not in st.session_state:
    st.session_state["model"] = None

# if "chat_history" not in st.session_state:
#     st.session_state["chat_history"] = []

if "active_conversation" not in st.session_state:
    st.session_state["active_conversation"] = None  # Track active conversation

if "model_change" not in st.session_state:
    st.session_state["model_change"] = False

if "topic" not in st.session_state:
    st.session_state["topic"] = None

if "continue" not in st.session_state:
    st.session_state["continue"] = False


# Define a function to query Ollama
def query_ollama(messages, model):
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={"model": model, "messages": messages, "stream": False}
        )
        if response.status_code == 200:
            return response.json().get("message", {}).get("content", "No response received")
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error connecting to Ollama: {e}"


# Define a function to query OpenAI models
def query_openai(messages, api_key):

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": messages
    }
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
        if response.status_code == 200:
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response received")
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error connecting to OpenAI: {e}"


def query_gemini(messages, api_key):

    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    response = client.chat.completions.create(
        model="gemini-1.5-flash-latest",
        n=1,
        messages=messages
    )

    return response.choices[0].message.content


def load_history():
    if st.session_state['active_conversation']:
        for m in st.session_state["messages"]:
            print(f"ch {m}")
            for r, c in m.items():
                if r == "role":
                    role = c
                elif r == "content":
                    if role == "user":
                        if isinstance(c, str) and not c.startswith("CONTEXT:"):
                            user_message = st.chat_message("user")
                            user_message.write(c)
                    elif role == "assistant":
                        assistant_message = st.chat_message("assistant", avatar="logo_t.png")
                        assistant_message.write(c)

def model_change():
    st.session_state["model_change"] = True

# File to store conversations persistently
CONVERSATION_FILE = "conversations.json"
# Load saved conversations from file if they exist
if os.path.exists(CONVERSATION_FILE):
    with open(CONVERSATION_FILE, "r") as f:
        st.session_state["all_conversations"] = json.load(f)
else:
    st.session_state["all_conversations"] = {}

st.logo("sanctum_t_l.png", size="large")

# st.sidebar.header("Setup Chat")

st.sidebar.subheader("Setup Conversation")

if st.session_state["database"]:
    st.sidebar.success(f"Connected to {st.session_state['database']}")
else:
    st.sidebar.error(f"Connect to DB above to begin !!!")

# Input for conversation topic
st.session_state["topic"] = st.sidebar.text_input("Enter Conversation Topic")

# st.sidebar.subheader("Choose a Model")
model_options = ["llama3.2", "OpenAI (GPT-3.5)", "codellama", "gemini-1.5-flash-latest"]
st.session_state["model"] = st.sidebar.selectbox("Select Model", options=model_options, index=1, on_change=model_change())

if st.session_state["model"] == "OpenAI (GPT-3.5)" or st.session_state["model"] == "gemini-1.5-flash-latest":
    st.session_state["OPEN_API_KEY"] = st.sidebar.text_input("Enter API Key (press enter)", type="password")

# Check if all inputs are filled to show the start button
st.session_state["chat_disabled"] = not st.session_state["database"] or not st.session_state["topic"] or not st.session_state["model"] or (st.session_state["model"] == "OpenAI (GPT-3.5)" and not st.session_state["OPEN_API_KEY"])

if "chat_button" in st.session_state and st.session_state.chat_button == True:
    st.session_state.running = True
    st.session_state.close = False
else:
    if not st.session_state["active_conversation"] and not st.session_state["continue"]:
        st.session_state.running = st.session_state["chat_disabled"]
        st.session_state.close = True

if st.session_state["active_conversation"]:
    # if st.session_state.close_button:
    if "close_button" in st.session_state and st.session_state.close_button == True:
        # Save the current conversation
        st.session_state["all_conversations"][st.session_state["active_conversation"]] = st.session_state[
            "messages"]

        # Persist conversations to file
        with open(CONVERSATION_FILE, "w") as f:
            json.dump(st.session_state["all_conversations"], f)

        st.session_state["active_conversation"] = None
        st.session_state["messages"] = []
        st.session_state.close = True
        st.session_state["topic"] = None
        st.session_state["model"] = None
        st.session_state["continue"] = False

uploaded_file = st.sidebar.file_uploader("Bring Your Own Data (Optional)", type=["csv"])

if uploaded_file is not None:
    upload_error = False
    if uploaded_file.name.endswith('.csv'):
        # Read CSV file
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        reader = csv.reader(stringio)
        content = "\n".join(["\t".join(row) for row in reader])
    else:
        st.error("Unsupported file type")
        upload_error = True

    if not upload_error:
        with st.expander("Data"):
            st.write(content)
        st.session_state["messages"].append({"role": "user", "content": f"CONTEXT: {content}"})


# Show Start Conversation button only if all fields are filled
left,middle,right,end = st.sidebar.columns(4)
chat_button = left.button("Start Chat", disabled=st.session_state.running , key="chat_button")
close_button = middle.button("Close Chat", disabled=st.session_state.close, key="close_button")

if st.session_state.get("chat_button") and st.session_state["topic"]:
    st.session_state["active_conversation"] = st.session_state["topic"]

    tables = ", ".join(st.session_state["schema"].keys())
    columns_info = "\n".join([f"Table `{table}`: {', '.join(st.session_state['schema'][table]['Field'])}" for table in st.session_state["schema"]])
    system_message = f"You are a SQL query assistant as part of Chat with DB. Available tables and their columns are:\n{columns_info}. Donot answer any query from user role other than SQL related"

    st.session_state["messages"].insert(0,{"role": "system", "content": system_message})

clear = []
with st.sidebar:
    st.subheader("Conversation History")
    # List all closed conversations
    if st.session_state["all_conversations"]:
        for topic, messages in st.session_state["all_conversations"].items():
            if "clear_"+topic not in st.session_state and "cont_"+topic not in st.session_state:
                with st.expander(topic):
                    for message in messages:
                        role = "You" if message["role"] == "user" else "Assistant"
                        timestamp = message.get("timestamp", "Unknown Time")
                        st.write(f"**{role}:** {message['content']}")

                    l,m,r,e = st.columns(4)
                    clear_button = m.button("Clear", key="clear_"+topic)
                    continue_button = r.button("Continue", key="cont_" + topic)
            else:
                if st.session_state["clear_"+topic] == False and st.session_state["cont_"+topic] == False:
                    with st.expander(topic):
                        for message in messages:
                            role = "You" if message["role"] == "user" else "Assistant"
                            timestamp = message.get("timestamp", "Unknown Time")
                            st.write(f"**{role}:** {message['content']}")

                        l, m, r, e = st.columns(4)
                        clear_button = m.button("Clear", key="clear_" + topic)
                        continue_button = r.button("Continue", key="cont_" + topic)
                else:
                    if st.session_state["cont_"+topic] == True:
                        st.session_state["continue"] = True
                        st.session_state["messages"] = st.session_state["all_conversations"][topic]
                        st.session_state["active_conversation"] = topic
                        st.session_state.running = True
                        st.session_state.close = False
                        st.session_state["model"] = None
                        clear.append(topic)
                    else:
                        clear.append(topic)

    from streamlit_extras.bottom_container import bottom

    with bottom():
        st.sidebar.write("&copy; 2025 Sanctum Digital Solutions")

for t in clear:
    del st.session_state["all_conversations"][t]
    # Persist conversations to file
    with open(CONVERSATION_FILE, "w") as f:
        json.dump(st.session_state["all_conversations"], f)


def stream_data(data):
    for w in data.split(" "):
        yield w + " "
        time.sleep(0.1)

chat = not st.session_state['active_conversation'] or st.session_state["model"] is None

# Header Section with Logo
col1, col2 = st.columns([1, 16])
with col1:
    st.image("logo_t.png", width=100)  # Replace 'path_to_logo.png' with the actual path to your logo image
with col2:
    st.title(":orange[Chat] with your DB")

if not st.session_state["active_conversation"]:
    st.info("Start a new conversation from the sidebar.")
else:
    st.write(f"### Topic: {st.session_state['active_conversation']}")
    st.session_state["chat_disabled"] = True

load_history()

# if chat_button:
nl_query = st.chat_input("Enter your query (e.g., 'show the total sales per product')", disabled=chat)

if nl_query:
    st.session_state["messages"].append({"role": "user", "content": nl_query})
    last = len(st.session_state["messages"]) - 1

    if st.session_state['active_conversation']:
        if st.session_state["model_change"]:
            for m in st.session_state["messages"][last:]:
                print(f"ch {m}")
                for r, c in m.items():
                    if r == "role":
                        role = c
                    elif r == "content":
                        if role == "user":
                            if isinstance(c, str) and not c.startswith("CONTEXT:"):
                                user_message = st.chat_message("user")
                                user_message.write(c)
                        elif role == "assistant":
                            assistant_message = st.chat_message("assistant", avatar="logo_t.png")
                            assistant_message.write(c)

    with st.spinner("Thinking..."):
        if st.session_state["model"]:
            if st.session_state["model"] == "OpenAI (GPT-3.5)":
                response = query_openai(st.session_state["messages"],st.session_state["OPEN_API_KEY"])
            elif st.session_state["model"] == "gemini-1.5-flash-latest":
                response = query_gemini(st.session_state["messages"], st.session_state["OPEN_API_KEY"])
            else:
                response = query_ollama(st.session_state["messages"], st.session_state["model"])

    response = response + "\n\n**model:" + st.session_state["model"]+"**"
    st.session_state["messages"].append({"role":"assistant", "content":response})
    assistant_message = st.chat_message("assistant", avatar="logo_t.png")
    assistant_message.write_stream(stream_data(response))


with bottom():
    st.write("**&copy; 2025 Sanctum Digital Solutions**")
