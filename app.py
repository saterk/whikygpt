import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.llms import LlamaCpp

@st.cache_resource
# def load_pipeline():
#     # Load local HF model
#     model_name = "tiiuae/falcon-7b-instruct"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name)
#     llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
#     return HuggingFacePipeline(pipeline=llm_pipeline)

def load_pipeline():
    return LlamaCpp(
        model_path="./models/phi-2.Q4_K_M.gguf",  # Adjust path if needed
        n_ctx=2048,
        n_threads=4,       # Tune for your CPU
        n_batch=8,
        temperature=0.7,
        top_p=0.9,
        verbose=True
    )

@st.cache_resource
def build_vectorstore():
    # Load and process docs
    loader = TextLoader("./data/whisky_notes.txt")
    raw_docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(raw_docs)

    # Embeddings & vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

def create_qa_chain(llm, vectorstore):
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a whisky expert assistant. Use the context below to answer the question.
You are a highly knowledgeable travel advisor. 
Your job is to provide travel information based on the context provided.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:
"""
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )

# Streamlit UI
st.set_page_config(page_title="Whisky Q&A RAG App", layout="wide")
st.title("🥃 Whisky Expert Q&A")

question = st.text_input("Ask a whisky-related question:")

if question:
    with st.spinner("Thinking..."):
        llm = load_pipeline()
        vectorstore = build_vectorstore()
        qa_chain = create_qa_chain(llm, vectorstore)

        result = qa_chain(question)
        st.markdown("### Answer")
        st.write(result['result'])

        with st.expander("📄 Sources"):
            for doc in result['source_documents']:
                st.markdown(doc.page_content)

#streamlit run app.py --server.port 8501
#ngrok config add-authtoken 2KAXyYA8Zcu4PGAE0NpQFf2K1jy_6NRkw3UXUaEPNJQ7pJZvX
#ngrok http 8501