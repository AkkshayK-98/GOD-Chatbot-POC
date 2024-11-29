import os
import base64
import tempfile
import streamlit as st
from vertexai import init
from vertexai.preview.generative_models import GenerativeModel, SafetySetting, Tool
from vertexai.preview.generative_models import grounding
import time

# Function to configure Google Cloud credentials from environment variable
def configure_google_credentials():
    encoded_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_BASE64")
    if not encoded_creds:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS_BASE64 environment variable is not set.")
    
    creds_json = base64.b64decode(encoded_creds).decode('utf-8')
    temp_creds_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    with open(temp_creds_file.name, "w") as f:
        f.write(creds_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_creds_file.name

# Function to initialize the chat model
def initialize_chat_model():
    init(project="god-chatbot-poc", location="us-central1")
    tools = [
        Tool.from_retrieval(
            retrieval=grounding.Retrieval(
                source=grounding.VertexAISearch(datastore="projects/god-chatbot-poc/locations/us/collections/default_collection/dataStores/poc-god-chatbot_1732787019287"),
            )
        ),
    ]
    model = GenerativeModel(
        "gemini-1.5-flash-002",
        tools=tools,
        system_instruction="""You have been provided some questions and answers about the Mahamantra. 
The questions are asked by devotees, and the answers are given by our Guru Maharaj. 
You will answer questions based on the given QAs between the devotees and Guru Maharaj. 
You DO NOT know anything other than this context. Under no circumstances are you to answer anything that is not related to the given context."""
    )
    return model.start_chat()

# Streamlit chatbot UI
def chat_ui():
    # Initialize session state for chat history and model
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []  # Chat history
    if "chat" not in st.session_state:
        configure_google_credentials()  # Configure Google credentials
        st.session_state["chat"] = initialize_chat_model()  # Initialize chat session

    st.title("GOD Chatbot - POC")
    st.markdown("Ask your questions about the Mahamantra below!")

    # Chat messages display
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input section
    if user_input := st.chat_input("Type your question here..."):
        # Add user input to chat history
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate bot response
        with st.chat_message("bot"):
            placeholder = st.empty()
            bot_response = generate_streaming_response(st.session_state["chat"], user_input, placeholder)

        # Add bot response to chat history
        st.session_state["chat_history"].append({"role": "bot", "content": bot_response})

# Stream the bot's response incrementally
def generate_streaming_response(chat, user_input, placeholder):
    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }
    safety_settings = [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
    ]
    response = chat.send_message(
        [user_input],
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    bot_response_parts = response.candidates[0].content.parts

    # Stream response dynamically
    full_response = ""
    for part in bot_response_parts:
        full_response += part.text
        placeholder.markdown(full_response)  # Update the bot response placeholder
        time.sleep(0.1)  # Simulate streaming delay for effect
    return full_response

# Run the Streamlit app
if __name__ == "__main__":
    chat_ui()
