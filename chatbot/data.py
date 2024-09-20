import os

import streamlit as st
from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from datasets import load_dataset
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder, PromptBuilder
from haystack.components.generators.chat import AzureOpenAIChatGenerator
from haystack.components.generators import AzureOpenAIGenerator
from haystack.components.joiners import BranchJoiner
from haystack.components.converters import OutputAdapter
from typing import List
from haystack.dataclasses import ChatMessage
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.components.writers import ChatMessageWriter
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
@st.cache_resource(show_spinner=False)
def download_test_data():
    # url = f"https://drive.google.com/drive/folders/uc?export=download&confirm=pbef&id={file_id}"
    url = "https://drive.google.com/drive/folders/1uDSAWtLvp1YPzfXUsK_v6DeWta16pq6y"
    with st.spinner(text="Downloading test data. This might take a minute."):
        # @TODO: replace gown solution with a custom solution compatible with GitHub and
        # use st.progress to get more verbose during download
        download_folder(url=url, quiet=False, use_cookies=False, output="./data/")


@st.cache_resource(show_spinner=False)
def load_data():
    print("start loading data")
    with st.spinner(text="Loading and indexing the provided dataset – hang tight! This may take a few seconds."):
        try:
            for key, value in st.secrets.items():
                os.environ[key] = value
        except FileNotFoundError as e:
            print(e)
            print("./streamlit/secrets.toml not found. Assuming secrets are already available" "as environmental variables...")

        # Memory components
        
        memory_store = InMemoryChatMessageStore()
        memory_retriever = ChatMessageRetriever(memory_store)
        memory_writer = ChatMessageWriter(memory_store)

        #load test documents
        # dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
        # docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

        # document_store = InMemoryDocumentStore()
        # document_store.write_documents(documents=docs)
        document_store = ChromaDocumentStore(persist_path="C:\\Users\\LAIW0\\OneDrive - Singapore Management University\\non-personal\\chatbot\\doc_store",distance_function="cosine")


        system_message = ChatMessage.from_system("You are a helpful AI assistant using provided supporting documents and conversation history to assist humans")

        user_message_template ="""Given the conversation history and the provided supporting documents, give a brief answer with step-by-step thoughts to the question.
        Note that supporting documents are not part of the conversation. If question can't be answered from supporting documents, say so without mentioning the supporting documents.

            Conversation history:
            {% for memory in memories %}
                {{ memory.content }}
            {% endfor %}

            Supporting documents:
            {% for doc in documents %}
                {{ doc.content }}
            {% endfor %}

            \nQuestion: {{query}}
            \nAnswer:
        """
        query_rephrase_template = """
        Rewrite the question for search while keeping its meaning and key terms intact.
        If the conversation history is empty, DO NOT change the query.
        Use conversation history only if necessary, and avoid extending the query with your own knowledge.
        If no changes are needed, output the current question as is.

        Conversation history:
        {% for memory in memories %}
            {{ memory.content }}
        {% endfor %}

        User Query: {{query}}
        Rewritten Query:
        """
        # query_rephrase_template = """
        #     Rewrite the question for search while keeping its meaning and key terms intact.
        #     If the conversation history is empty, DO NOT change the query.
        #     Use conversation history to provide context only if necessary, and avoid extending the query with your own knowledge.
        #     If you find some necessary details are ignored, add it to make the query more plausible according to the related text.
        #     If no changes are needed, output the current question as is.

        #     Conversation history:
        #     {% for memory in memories %}
        #         {{ memory.content }}
        #     {% endfor %}

        #     User Query: {{query}}
        #     Rewritten Query:
        # """
        user_message = ChatMessage.from_user(user_message_template)
        conversational_rag = Pipeline()
        st.session_state["user_message"] = user_message
        st.session_state["system_message"] = system_message

        # components for query rephrasing
        conversational_rag.add_component("query_rephrase_prompt_builder", PromptBuilder(query_rephrase_template))
        conversational_rag.add_component("query_rephrase_llm", AzureOpenAIGenerator())
        conversational_rag.add_component("list_to_str_adapter", OutputAdapter(template="{{ replies[0] }}", output_type=str))
        # components for query rephrasing
        retriever = ChromaEmbeddingRetriever(document_store=document_store, top_k=5)
        text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2") 
        conversational_rag.add_component("text_embedder", text_embedder)
        conversational_rag.add_component("retriever", retriever)
        conversational_rag.add_component("prompt_builder", ChatPromptBuilder(variables=["query", "documents", "memories"], required_variables=["query", "documents", "memories"]))
        conversational_rag.add_component("llm", AzureOpenAIChatGenerator())

        # components for memory
        conversational_rag.add_component("memory_retriever", ChatMessageRetriever(memory_store))
        conversational_rag.add_component("memory_writer", ChatMessageWriter(memory_store))
        conversational_rag.add_component("memory_joiner", BranchJoiner(List[ChatMessage]))

        # connections for query rephrasing
        conversational_rag.connect("memory_retriever", "query_rephrase_prompt_builder.memories")
        conversational_rag.connect("query_rephrase_prompt_builder.prompt", "query_rephrase_llm")
        conversational_rag.connect("query_rephrase_llm.replies", "list_to_str_adapter")
        conversational_rag.connect("list_to_str_adapter", "text_embedder")
        conversational_rag.connect("text_embedder", "retriever.query_embedding")

        # connections for RAG
        conversational_rag.connect("retriever", "prompt_builder.documents")
        conversational_rag.connect("prompt_builder.prompt", "llm.messages")
        conversational_rag.connect("llm.replies", "memory_joiner")

        # connections for memory
        conversational_rag.connect("memory_joiner", "memory_writer")
        conversational_rag.connect("memory_retriever", "prompt_builder.memories")
        return conversational_rag,memory_store