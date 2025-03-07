from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
# File path to the PDF
FILE_PATH = "../documents/Human_Resource_Management.pdf"

# Create a loader
loader = PyPDFLoader(FILE_PATH)

# Load pages from the PDF
pages = loader.load_and_split()

# Extract text content from each page
# pdf_text = [doc.page_content for doc in pages]

# Print extracted pages
print(f"Total pages loaded: {len(pages)}")

# Load embedding model
embedding_function = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")  # Ensure correct model name

# Create a Chroma vector store (pass the embedding function, not precomputed embeddings)
vector_db = Chroma.from_documents(
    documents = pages,
    embedding = embedding_function,  # ✅ Pass model, not embeddings
    persist_directory="../vector_db1",  # Ensure this directory exists
    collection_name="HR_resources_system"
)

# Persist the database
vector_db.persist()

print("Vector database created successfully!")

#make persistant


