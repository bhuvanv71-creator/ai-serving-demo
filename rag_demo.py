from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter

# Sample documents (in real use, this would be your company's docs, PDFs, etc.)
documents = [
    "DevOps combines software development and IT operations to shorten the development lifecycle.",
    "Kubernetes is a container orchestration platform that automates deployment, scaling, and management.",
    "MLOps applies DevOps principles to machine learning, covering model training, deployment, and monitoring.",
    "Terraform is an infrastructure-as-code tool used to provision and manage cloud resources.",
    "RAG stands for Retrieval-Augmented Generation, combining a knowledge base with an LLM's responses."
]

# Split text into chunks (needed for larger real documents)
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
docs = splitter.create_documents(documents)

# Convert text into vector embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store embeddings in a local vector database
vectorstore = Chroma.from_documents(docs, embeddings)

# Simulate a user question
query = "What is Kubernetes used for?"
results = vectorstore.similarity_search(query, k=2)

print(f"\nQuery: {query}\n")
print("Top matching documents:")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}")