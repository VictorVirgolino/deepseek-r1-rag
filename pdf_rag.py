from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from streamlit import streamlit as st

pdfs_directory = "./pdfs/"

embeddings = OllamaEmbeddings(model="deepseek-r1:14b")

model = OllamaLLM(model="deepseek-r1:14b", temperature=0.3)

vector_store = InMemoryVectorStore(embeddings)

chunk_size = 2000
chunk_overlap = 400
add_start_index = True

template = """
Você é um assistente para tarefas de perguntas e respostas. Use os seguintes trechos de contexto recuperado para responder à pergunta. 
Se você não souber a resposta, apenas diga que não sabe. Use no máximo três frases e mantenha a resposta concisa.
Sempre Responda em Português.
Pergunta: {pergunta}
Contexto: {contexto}
Resposta:
"""


def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())


def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()

    return documents


def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=add_start_index,
    )

    docs = text_splitter.split_documents(documents)
    return docs


def index_docs(documents):
    vector_store.add_documents(documents)


def retrieve_docs(query):
    result = vector_store.similarity_search(query, k=7)
    return result


def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    answer = chain.invoke({"pergunta": question, "contexto": context})
    return answer


uploaded_file = st.file_uploader(
    "Upload PDF", type=["pdf"], accept_multiple_files=False
)

if uploaded_file:
    upload_pdf(uploaded_file)
    docs = load_pdf(pdfs_directory + uploaded_file.name)
    chunked_docs = split_text(docs)
    question = st.chat_input("Ask a Question")

    if question:
        related_docs = retrieve_docs(question)
        answer = answer_question(question, related_docs)
        st.chat_message("assistant").write(answer)