# chain.py
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .retriever import LangChainQdrantRetriever
from models.settings import settings
from pydantic import BaseModel
from langchain_together import Together


def format_docs(docs):
    """Format the retrieved documents into a string for LLM input."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(retriever: LangChainQdrantRetriever):
    """Create the RAG chain using the provided retriever."""
    if retriever is None:
        raise ValueError("Retriever cannot be None")
    llm = Together(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    together_api_key=settings.TOGETHER_API_KEY
)
    # Pull the RAG prompt from langchain hub
    prompt = hub.pull("rlm/rag-prompt")

    # Initialize LLM
    # llm = TogetherLLM()

    # Define the chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
