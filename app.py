import os
import json
import base64
import tempfile
import streamlit as st
import vertexai
from vertexai.preview.generative_models import GenerativeModel, SafetySetting, Tool
from vertexai.preview.generative_models import grounding

# Function to configure Google Cloud credentials from environment variable
def configure_google_credentials():
    encoded_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_BASE64")
    if not encoded_creds:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS_BASE64 environment variable is not set.")
    
    # Decode the base64 string to get the original JSON credentials
    creds_json = base64.b64decode(encoded_creds).decode('utf-8')
    
    # Write the credentials JSON to a temporary file
    temp_creds_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    with open(temp_creds_file.name, "w") as f:
        f.write(creds_json)

    # Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the file path
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_creds_file.name

# Function to initialize the chat model
def initialize_chat_model():
    vertexai.init(project="god-chatbot-poc", location="us-central1")
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

# Streamlit UI for chatbot
def chat_ui():
    st.title("Maha Mantra Chatbot")

    user_input = st.text_input("Ask a question:")
    if st.button("Send") and user_input:
        try:
            # Call the credentials setup function
            configure_google_credentials()

            # Initialize chat session
            chat = initialize_chat_model()

            # Send user message to the chat model and get response
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
            bot_response = response.candidates[0].content.parts[0].text
            st.write(f"Chatbot: {bot_response}")

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Run the Streamlit UI
if __name__ == "__main__":
    chat_ui()
