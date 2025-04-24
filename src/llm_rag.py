from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from transformers import pipeline
import os
from glob import glob

def setup_rag(knowledge_base_path, index_path="data/faiss_index"):
    """
    Set up RAG by indexing documents in the knowledge base.
    Args:
        knowledge_base_path (str): Path to a text file or directory with .txt files.
        index_path (str): Path to save/load FAISS index.
    Returns:
        FAISS: Vector store with indexed documents.
    """
    # Check if FAISS index exists
    if os.path.exists(index_path):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        return db

    # Load documents
    documents = []
    if os.path.isdir(knowledge_base_path):
        for file_path in glob(os.path.join(knowledge_base_path, "*.txt")):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
    elif os.path.isfile(knowledge_base_path):
        loader = TextLoader(knowledge_base_path)
        documents = loader.load()
    else:
        raise ValueError(f"{knowledge_base_path} is not a valid file or directory")

    if not documents:
        raise ValueError("No documents found in the knowledge base")

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Index documents in FAISS
    db = FAISS.from_documents(docs, embeddings)

    # Save FAISS index
    db.save_local(index_path)
    return db

def generate_strategy(db, query, llm_model="gpt2", max_new_tokens=50):
    """
    Generate a trading strategy using retrieved documents and an LLM.
    Args:
        db (FAISS): Indexed vector store.
        query (str): User query (e.g., "Generate a trading strategy for AAPL").
        llm_model (str): Hugging Face model name.
        max_new_tokens (int): Number of new tokens to generate.
    Returns:
        str: Generated trading strategy.
    """
    # Initialize LLM
    llm = pipeline("text-generation", model=llm_model)

    # Retrieve relevant documents
    retrieved_docs = db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Create prompt
    prompt = (
        f"You are a professional stock trading expert with deep knowledge of technical analysis and risk management. "
        f"Using the following information, generate a detailed trading strategy for the specified stock. "
        f"The strategy should include:\n"
        f"- Entry rules (when to buy, including specific technical indicators).\n"
        f"- Exit rules (when to sell, including take-profit and stop-loss).\n"
        f"- Risk management (position sizing, risk per trade, risk-reward ratio).\n"
        f"- Suitability (market conditions or stock characteristics).\n"
        f"Ensure the strategy is clear, actionable, and suitable for trading on Webull.\n\n"
        f"Information:\n{context}\n\nQuery: {query}"
    )

    # Generate strategy
    response = llm(prompt, max_new_tokens=max_new_tokens, num_return_sequences=1, truncation=True)[0]["generated_text"]
    return response

if __name__ == "__main__":
    # Example usage
    db = setup_rag("data/knowledge_base/")
    query = "Generate a stock trading strategy for AAPL using technical indicators."
    strategy = generate_strategy(db, query)
    print("Generated Strategy:\n", strategy)