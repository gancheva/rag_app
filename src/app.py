import re
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub


load_dotenv()  # load the HUGGINGFACEHUB_API_TOKEN from the .env file


def get_pdf_text(pdf_docs):
    """
    Extract the text from pdf documents.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """
    Split the text in chunks/splits.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    """
    Vectorize the text chunks so that similarity search can be efficiently performed.
    """
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    """
    Create a conversational chain with Flau-UL2 LLM, incorporate memory, and utilize the vector store as a data retriever.
    """    
    llm = HuggingFaceHub(repo_id="google/flan-ul2", model_kwargs={"temperature":0.1, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def transform_conversation(conversation_string):
    """
    Reformat the conversation/chat history so that it can be displayed as a chat on the screen using Streamlit.
    """
    pattern = r'(AI|Human):\s(.+?)(?=(AI|Human):|$)'
    matches = re.findall(pattern, conversation_string)
    json_objects = []
    for match in matches:
        role, content = match[0], match[1]
        json_obj = {'role': role, 'content': content}
        json_objects.append(json_obj)

    return json_objects            


def clear_chat_history():
    """
    Initialize a new chat history.
    """
    st.session_state.chat_history = "AI: Hello there! How can I halp you today?"


def process_uploaded_files(pdf_docs):
    """
    If no documents are uploaded, set everything to default values. 
    Otherwise, initialize the vector store and the conversation chain.
    """
    if pdf_docs is None:
        st.session_state.vectorstore = None
        st.session_state.conversation = None
        st.markdown('<style> div.uploadedFile { display: none; } </style>', unsafe_allow_html=True)
    else:
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        st.session_state.vectorstore = get_vectorstore(text_chunks)
        st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)


st.set_page_config(page_title="Chat with multiple PDFs",
                    page_icon=":books:")
st.markdown('<style> ol { margin: auto; } h3 { margin-top: 1rem; } </style>', unsafe_allow_html=True)

if "conversation" not in st.session_state.keys():
    st.session_state.conversation = None
if "vectorstore" not in st.session_state.keys():
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state.keys():
    clear_chat_history()

with st.sidebar:
    st.subheader("📖 RULES FOR USING THE AI") 
    st.markdown('1. Upload your PDF file/files')
    st.markdown('2. Press the button "Process Uploaded Files"')
    st.markdown('3. Afther the processing finishes, ask your question')
 
    st.subheader("CHAT WITH YOUR DOCUMENTS") 
    pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
    
    if st.button("Process Uploaded Files"):
        placeholder = st.empty()
        with placeholder, st.spinner("Processing"):
            process_uploaded_files(pdf_docs)
    st.empty()

    if st.button('Clear Chat History'):
        clear_chat_history()
    
    if st.button('Delete Uploaded Files'):
        process_uploaded_files(None)

    st.subheader("DISCLAIMER") 
    st.markdown('Please note that this application is designed solely for demonstration purposes, showcasing the capabilities of open source AI technology. Any documents uploaded are temporarily stored solely for the purpose of addressing related inquiries.')

for message in transform_conversation(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input(disabled=st.session_state.conversation is None)
if prompt:
    st.session_state.chat_history += f"Human: {prompt}"
    with st.chat_message("user"):
        st.write(prompt)
  
if st.session_state.chat_history.rsplit(":", 1)[0].endswith("Human"):
    full_response = ''
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."): 
            response = st.session_state.conversation({
                'question': prompt,
                'chat_history': st.session_state.chat_history
            })
            placeholder = st.empty()
            full_response = response['answer'] if 'answer' in response.keys() else 'No answer can be given'
            placeholder.markdown(full_response)
    st.session_state.chat_history += f"AI: {full_response}"

