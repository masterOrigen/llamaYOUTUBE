import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader, ChatPromptTemplate
from llama_index.llms.groq import Groq
from llama_index.embeddings.groq import GroqEmbedding
from llama_index.core import Settings
from youtube_transcript_api import YouTubeTranscriptApi
import shutil
import os
import time

icons = {"assistant": "robot.png", "user": "man-kddi.png"}

# Lista de modelos de Groq disponibles
groq_models = [
    "llama2-70b-4096",
    "mixtral-8x7b-32768",
    "gemma-7b-it",
]

# Streamlit app initialization
st.title("Chat with your PDF📄")
st.markdown("**Built by [Pachaiappan❤️](https://mr-vicky-01.github.io/Portfolio/)**")

# Selector de modelos de Groq en la barra lateral
with st.sidebar:
    st.title("Menu:")
    selected_model = st.selectbox("Select Groq Model", groq_models)

    uploaded_file = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button")
    video_url = st.text_input("Enter Youtube Video Link: ")

# Configure the Llama index settings
@st.cache_resource
def get_llm(model_name):
    return Groq(
        model_name=model_name,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
    )

Settings.llm = get_llm(selected_model)
Settings.embed_model = GroqEmbedding(
    model_name="llama2-70b-4096",
    api_key=os.getenv("GROQ_API_KEY")
)

# Define the directory for persistent storage and data
PERSIST_DIR = "./db"
DATA_DIR = "data"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

def data_ingestion():
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

def remove_old_files():
    directory_path = "data"
    shutil.rmtree(directory_path)
    os.makedirs(directory_path)

def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join(i["text"] for i in transcript_text)
        return transcript
    except Exception as e:
        st.error(e)

def handle_query(query):
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    chat_text_qa_msgs = [
    (
        "user",
        """You are Q&A assistant named CHATTO, created by Pachaiappan [linkdin](https://www.linkedin.com/in/pachaiappan) an AI Specialist. Your main goal is to provide answers as accurately as possible, based on the instructions and context you have been given. If a question does not match the provided context or is outside the scope of the document, you only say the user to 'Please ask a questions within the context of the document'.
        Context:
        {context_str}
        Question:
        {query_str}
        """
    )
    ]
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
    query_engine = index.as_query_engine(text_qa_template=text_qa_template)
    answer = query_engine.query(query)
    
    if hasattr(answer, 'response'):
        return answer.response
    elif isinstance(answer, dict) and 'response' in answer:
        return answer['response']
    else:
        return "Sorry, I couldn't find an answer."

def streamer(text):
    for i in text:
        yield i
        time.sleep(0.001)

# Continuación del código de la interfaz de Streamlit
if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', "content": 'Hello! Upload a PDF/Youtube Video link and ask me anything about the content.'}]

for message in st.session_state.messages:
    with st.chat_message(message['role'], avatar=icons[message['role']]):
        st.write(message['content'])

with st.sidebar:
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            if len(os.listdir("data")) != 0:
                remove_old_files()
                
            if uploaded_file:
                filepath = "data/saved_pdf.pdf"
                with open(filepath, "wb") as f:
                    f.write(uploaded_file.getbuffer())
        
            if video_url:
                extracted_text = extract_transcript_details(video_url)
                with open("data/saved_text.txt", "w") as file:
                    file.write(extracted_text)
                
            data_ingestion()  # Process PDF every time new file is uploaded
            st.success("Done")

user_prompt = st.chat_input("Ask me anything about the content of the PDF:")

if user_prompt and (uploaded_file or video_url):
    st.session_state.messages.append({'role': 'user', "content": user_prompt})
    with st.chat_message("user", avatar="man-kddi.png"):
        st.write(user_prompt)

    # Trigger assistant's response retrieval and update UI
    with st.spinner("Thinking..."):
        response = handle_query(user_prompt)
    with st.chat_message("assistant", avatar="robot.png"):
        st.write_stream(streamer(response))
    st.session_state.messages.append({'role': 'assistant', "content": response})
