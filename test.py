import os
OPENAI_API_TYPE = "azure"
OPENAI_API_VERSION = "2023-05-15"
OPENAI_API_BASE = "https://azureopenaiservice-eastus.openai.azure.com/openai"
OPENAI_API_KEY = "e52a36482c1e413e9d0f501e498f8075"
DEPLOYMENT_NAME = "pdf-reader"

from dotenv import load_dotenv

os.environ["OPENAI_API_TYPE"] = OPENAI_API_TYPE
os.environ["OPENAI_API_VERSION"] = OPENAI_API_VERSION
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

load_dotenv()

import streamlit as st
import tempfile

st.title("PDF Promt based QnA")

uploadedFile = st.file_uploader("Upload a file", type=["pdf"])

if uploadedFile is not None:
    temp = tempfile.NamedTemporaryFile(mode="wb", delete=False)
    bytes_data = uploadedFile.getvalue()
    temp.write(bytes_data)
    fileName = temp.name

    from langchain.document_loaders import UnstructuredFileLoader

    loader = UnstructuredFileLoader(fileName)
    documents = loader.load()

    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.chains import RetrievalQA
    from langchain.llms import AzureOpenAI

    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
    )

    doc_search = Chroma.from_documents(documents=texts,embedding=embeddings)
    chain = RetrievalQA.from_chain_type(llm=AzureOpenAI(deployment_name = DEPLOYMENT_NAME,model_name='gpt-35-turbo-instruct'),chain_type='stuff',retriever = doc_search.as_retriever())

    import fitz

    with fitz.open(fileName) as doc:
        context = ""
        for page in doc:
            context += page.get_text()

    query = st.text_input("Enter a prompt:")

    output = chain.run(query=query,context=context)

    # st.write(type(query))
    st.write(output)
