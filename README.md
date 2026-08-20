Hybrid RAG Chatbot
A Retrieval-Augmented Generation (RAG) chatbot built with LangChain, Pinecone, OpenAI GPT-4o, and Streamlit. This application allows users to ask questions against a document knowledge base and generates answers using relevant information retrieved from the indexed documents.
The project uses a hybrid retrieval approach during document ingestion and retrieval to improve the relevance of results compared with relying solely on traditional vector similarity search.

Summary:
This project combines hybrid retrieval with large language model generation to create a document-aware conversational application.
ingestion_hybrid.py is responsible for transforming the document knowledge base into a searchable vector representation in Pinecone, while chatbot_hybrid_rag.py handles user queries, retrieves and reranks relevant information, constructs the context, and uses GPT-4o to generate the final response.

```text
              DOCUMENT KNOWLEDGE BASE
                       │
                       ▼
              ingestion_hybrid.py
                       │
                       ▼
               Hybrid Embeddings
                       │
                       ▼
                   Pinecone
                       │
                       │
                       ▼
              chatbot_hybrid_rag.py
                       │
                 User Question
                       │
                       ▼
                Query Embedding
                       │
                       ▼
              Similarity Retrieval
                       │
                       ▼
                    Reranking
                       │
                       ▼
              Context Construction
                       │
                       ▼
                   OpenAI GPT-4o
                       │
                       ▼
                 Final Response

```

Architecture
The application consists of two primary Python scripts:

```text
Documents
    │
    ▼
┌─────────────────────────┐
│  ingestion_hybrid.py    │
│                         │
│  • Load documents       │
│  • Generate embeddings  │
│  • Create/index vectors │
└────────────┬────────────┘
             │
             ▼
       ┌───────────┐
       │ Pinecone  │
       │ Vector DB │
       └─────┬─────┘
             │
             ▼
┌─────────────────────────┐
│ chatbot_hybrid_rag.py   │
│                         │
│  • Accept user query    │
│  • Generate embeddings  │
│  • Retrieve documents   │
│  • Rerank documents     │
│  • Build context        │
│  • Generate answer      │
└────────────┬────────────┘
             │
             ▼
       ┌─────────────┐
       │  OpenAI     │
       │  GPT-4o     │
       └──────┬──────┘
              │
              ▼
        Final Answer
```

How to run?
Modify api credentials in the .env file
create docker image using "docker_file"
--  docker build -f docker_file -t hybrid-rag-chatbot .
Run docker image
--docker run --env-file .env -p 8501:8501 hybrid-rag-chatbot

open http://localhost:8501 or API endpoint in the cloud

