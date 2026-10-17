from langchain.chat_models import init_chat_model   #LLM
from langchain.embeddings import init_embeddings    #Embedding
from langchain_core.vectorstores import InMemoryVectorStore  #VectorStore
from langchain_community.document_loaders import TextLoader, PyPDFLoader  #DocumentLoader
from langchain_text_splitter import RecursiveCharacterTextSplitter  #TextSplitter

chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)
vs = InMemoryVectorStore(embeddings=init_embeddings(model="gemma3:1b", model_provider="ollama"))
llm = init_chat_model(model="gemma3:1b", model_provider="ollama")

cxt = ""

#Step 1: Retrieval
#Step 2: Generation