import os

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


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

from collections.abc import Sequence

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


    # if __name__ == '__main__':  
    #     st.set_option('server.enableCORS', True)

# import streamlit as st
# st.title("Hello, World!")


# Libraries:
    
# streamlit; python_version == '3.10'
# pysqlite3-binary; python_version == '3.10'
# python-dotenv; python_version == '3.10'
# langchain; python_version == '3.10'
# unstructured==0.7.12; python_version == '3.10'
# unstructured[pdf]; python_version == '3.10'
# openai; python_version == '3.10'
# chromadb; python_version == '3.10'
# fitz; python_version == '3.10'
# tiktoken; python_version == '3.10'
# pathlib2; python_version == '3.10'
    
# pip3 install -r requirements.txt
# python streamlit run test.py --server.port 8000 --server.address 0.0.0.0