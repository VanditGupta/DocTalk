import tempfile
from PIL import Image
import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
from dotenv import load_dotenv
load_dotenv()

# Set the title and subtitle of the app
# st.title('DocTalk: Unleash the Power of Conversational Documents')
st.set_page_config(layout="wide")

col1_width = 100
col2_width = 900

# Create a columns layout
col1, col2 = st.columns([col1_width, col2_width])

# Load and display the image in the left column
with col1:
    st.markdown("<br>", unsafe_allow_html=True)
    your_image = Image.open('CHAT WITH DOC.png')
    st.image(your_image, use_column_width=True, width = 100)

# Display the text in the right column
with col2:
    #st.title('DocTalk: Unleash the Power of Conversational Documents',  anchor='center')
    st.markdown(
        "<h1 style='max-width: 100%;'>DocTalk: Unleash the Power of Conversational Documents</h1>", 
        unsafe_allow_html=True
    )
    # st.markdown("<br>", unsafe_allow_html=True)
    # Upload PDF file
    st.subheader('Load your PDF, ask questions, and receive answers directly from the document.')
    st.subheader('Upload your PDF')
    uploaded_file = st.file_uploader('Choose a PDF file', type=(['pdf']))

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        st.success("PDF file uploaded successfully!")
        st.info("Full path of the uploaded file: {}".format(temp_file_path))

        # Set API key for OpenAI Service
        # Can be replaced with other LLM providers
        openai_api_key  = os.getenv('OPENAI_API_KEY')

        # Create instance of OpenAI LLM
        llm = OpenAI(temperature=0.1, verbose=True)
        embeddings = OpenAIEmbeddings()

        # Create and load PDF Loader
        loader = PyPDFLoader(temp_file_path)
        # Split pages from pdf 
        pages = loader.load_and_split()

        # Load documents into vector database aka ChromaDB
        store = Chroma.from_documents(pages, embeddings, collection_name='Pdf')

        # Create vectorstore info object
        vectorstore_info = VectorStoreInfo(
            name="Pdf",
            description="A PDF file to answer your questions",
            vectorstore=store
        )

        # Convert the document store into a langchain toolkit
        toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

        # Add the toolkit to an end-to-end LC
        agent_executor = create_vectorstore_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True
        )

        # Create a text input box for the user
        prompt = st.text_input('Input your prompt here')

        # If the user hits enter
        if prompt:
            # Then pass the prompt to the LLM
            response = agent_executor.run(prompt)
            # Write it out to the screen
            st.subheader('Answer:')
            st.write(response)

            # With a streamlit expander  
            with st.expander('Document Similarity Search'):
                # Find the relevant pages
                search = store.similarity_search_with_score(prompt) 
                # Write out the first 
                st.subheader('Top Similar Page:')
                st.write(search[0][0].page_content)
