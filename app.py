import tempfile
import streamlit as st
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from secret_key import openai_api_key

# Directory path for storing the Chroma database
DIR_PATH = "data"
openai_api_key = ""  # Your OpenAI API key

# List of keywords related to agriculture
agriculture_keywords = ["crop", "soil", "irrigation", "pest", "harvest", "farm", "agriculture", "fertilizer", "yield", "plant",
                       "remedies","symptoms","causes"]

# Process uploaded documents
def process_docs(uploads):
    documents = []
    for file in uploads:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            filename = file.name
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(tmp_file.name)
                documents.extend(loader.load())
            elif filename.endswith('.docx') or filename.endswith('.doc'):
                loader = Docx2txtLoader(tmp_file.name)
                documents.extend(loader.load())
            elif filename.endswith('.txt'):
                loader = TextLoader(tmp_file.name)
                documents.extend(loader.load())
    return documents

# Generate a response based on the prompt, texts, embeddings, and chat history
def generate_response(prompt, texts, embeddings, chat_history):
    db = Chroma.from_documents(texts, embedding=embeddings, persist_directory=DIR_PATH)
    db.persist()
    retriever = db.as_retriever(search_kwargs={'k': 7})

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model=model, openai_api_key=openai_api_key),
        retriever=retriever,
        return_source_documents=True
    )
    response = qa_chain.invoke({'question': prompt, 'chat_history': chat_history})
    return response

# Check if the prompt is related to agriculture
def is_agriculture_related(prompt):
    return any(keyword in prompt.lower() for keyword in agriculture_keywords)

# Initialize session state keys if they don't exist
if 'texts' not in st.session_state:
    st.session_state['texts'] = []
if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = None
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []
if 'file_uploader_key' not in st.session_state:
    st.session_state['file_uploader_key'] = 0

# Main Streamlit app
st.title("Document-Based Conversational AI")

uploaded_files = st.file_uploader("Upload your agricultural documents", accept_multiple_files=True, key=st.session_state['file_uploader_key'])
upload_button = st.button("Upload")
clear_button = st.button("Clear")

if upload_button:
    with st.spinner('Uploading...'):
        if uploaded_files:
            documents = process_docs(uploaded_files)
            text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=10)
            texts = text_splitter.split_documents(documents)
            st.session_state['texts'] = texts
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            st.session_state['embeddings'] = embeddings
    if embeddings:
        st.sidebar.write("Uploading done")

# Map model names to OpenAI model IDs
model_name = st.sidebar.selectbox("Select Model", ["GPT-3.5", "GPT-4"])
model = "gpt-3.5-turbo" if model_name == "GPT-3.5" else "gpt-4"

# Reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = []
    st.session_state['model_name'] = []
    st.session_state['texts'] = []
    st.session_state['embeddings'] = None
    st.session_state["file_uploader_key"] += 1
    st.experimental_rerun()

# Container for chat history
response_container = st.container()
# Container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        if st.session_state.get('embeddings'):
            if is_agriculture_related(user_input):
                output = generate_response(user_input, st.session_state['texts'], st.session_state['embeddings'], st.session_state['messages'])
            else:
                output = {"answer": "This query does not appear to be related to agriculture. Please ask an agriculture-related question."}
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output['answer'])
            st.session_state['messages'].append((user_input, output['answer']))
            st.session_state['model_name'].append(model_name)
        else:
            st.warning('No files uploaded', icon="⚠️")

if st.session_state.get('generated'):
    with response_container:
        for i in range(len(st.session_state['generated'])):
            st.text_area("User:", st.session_state["past"][i], key=str(i) + '_user')
            st.text_area("Bot:", st.session_state["generated"][i], key=str(i) + '_bot')