import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import os

#Load API Keys 
load_dotenv()
os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# DuckDuckGo Search 
search = DuckDuckGoSearchRun()

# HuggingFace Embeddings 
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Google Gemini LLM 
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def process_topic_and_query(topic, query):
    #Use DuckDuckGo to get content
    try:
        search_content = search.run(topic)
        if not search_content or len(search_content.strip()) == 0:
            return "No search results found. Please try a different topic."
    except Exception as e:
        return f"Search failed: {str(e)}"

    #Save content to AI.txt
    with open("AI.txt", "w", encoding="utf-8") as f:
        f.write(search_content)

    #Load and split documents
    loader = TextLoader("AI.txt", encoding="utf-8")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    #Create vector store with chunks
    try:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            collection_name="example_collection"
        )
    except Exception as e:
        return f"Embedding failed: {str(e)}. Please check your HuggingFace setup."

    #Semantic search for query
    results = vector_store.similarity_search(query, k=2)
    content = "\n\n".join([doc.page_content for doc in results])

    # Prompt LLM
    prompt = PromptTemplate(
        template=(
            "Here is the content: {content}\n\n"
            "Here is the query: {query}\n\n"
            "Respond ONLY based on the content. "
            "If the data is missing in the content, say: 'I don't know'."
        ),
        input_variables=["content", "query"]
    )
    parser = StrOutputParser()
    chain = prompt | llm | parser

    return chain.invoke({"query": query, "content": content})

#Streamlit App 
st.title("AI Web Search & Summary App")
st.markdown("Enter a topic to fetch content from the web and summarize it using LLM.")

topic = st.text_input("Topic")
query = st.text_input("Query")

if st.button("Generate Summary"):
    if not topic or not query:
        st.warning("Please enter both topic and query.")
    else:
        with st.spinner("Fetching, processing, and generating summary..."):
            result = process_topic_and_query(topic, query)
        st.success("Summary generated!")
        st.markdown("### Result:")
        st.write(result)