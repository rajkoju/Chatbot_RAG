# import basics
import os
import time
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

#documents
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv() 
bm25 = BM25Encoder().default()

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# initialize pinecone database
index_name = os.environ.get("PINECONE_INDEX_NAME")  # change if desired

# check whether index exists, and create if not
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="dotproduct", #cosine
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)




index = pc.Index(index_name)

# initialize embeddings model + vector store
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large",api_key=os.environ.get("OPENAI_API_KEY"))

#embeddings = OpenAIEmbeddings(model="text-embedding-3-large",api_key=os.environ.get("OPENAI_API_KEY"))

#embeddigs = embedding_model.embed([chunk.page_content for chunk in documents])
#dense_vec = embeddings.embed_documents(text_chunks)
#dense_vec = embedding_model.embed([chunk.page_content for chunk in documents])


#print(len(embedding.data[0].embedding))

#vector_store = PineconeVectorStore(index=index, embedding=embeddings)




# loading the PDF document
loader = PyPDFDirectoryLoader("documents/")

raw_documents = loader.load()

# splitting the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=400,
    length_function=len,
    is_separator_regex=False,
)

#chunk_size=800
#chunk_overlap=400
#
#This is good for research papers, but many production systems use:
#
#chunk_size = 400–600
#chunk_overlap = 100–200
#because smaller chunks often improve retrieval precision.


# creating the chunks
documents = text_splitter.split_documents(raw_documents)
#dense_vec = embedding_model.embed([chunk.page_content for chunk in documents])

texts = [chunk.page_content for chunk in documents]
bm25.fit(texts)
bm25.dump("bm25.json")
sparse_vecs = bm25.encode_documents(texts)
dense_vecs = embeddings_model.embed_documents(texts)

records = []

for i, doc in enumerate(documents):
    records.append({
        "id": str(i),
        "values": dense_vecs[i],
        "sparse_values": sparse_vecs[i],
        #"metadata": {"text": text}
        "metadata":{ "text": doc.page_content,
            **doc.metadata
     }
    })

print(records[-1])
#index.upsert(records)

batch_size = 100

for i in range(0, len(records), batch_size):
    index.upsert(records[i:i+batch_size])

# generate unique id's

#i = 0
#uuids = []
#while i < len(documents):
#    i += 1
#    uuids.append(f"id{i}")



first_text = documents[0].page_content
first_embedding = embeddings_model.embed_query(first_text)
print("Embedding vector length:", len(first_embedding))
print("First 10 values:", first_embedding[:10])
print( first_embedding)

#Side-by-side comparison
#Function	Library	Runs where	Vector size
#embed_query()	LangChain + OpenAI	OpenAI API	3072
#embed_documents()	LangChain + OpenAI	OpenAI API	3072
#encode()	Sentence Transformers	Local machine	384
#Key difference
#Function	Input	Output
#embed_query()	single string	single vector
#embed_documents()	list of strings	list of vectors


# add to database

#vector_store.add_documents(documents=documents, ids=uuids)
#When langchain does vector_store.add_documents(documents=documents), it actually does dense_vec = embeddings.embed_documents(text_chunks)
