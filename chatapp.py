import streamlit as st
from openai import OpenAI
import os
from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from datasets import load_dataset
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.components.writers import ChatMessageWriter
from haystack.dataclasses import ChatMessage

# from chatbot.data import download_test_data
from chatbot.data import load_data


    
st.title("RAT app test")
conversational_rag , memory_store = load_data()
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # stream = client.chat.completions.create(
        #     model=st.session_state["openai_model"],
        #     messages=[
        #         {"role": m["role"], "content": m["content"]}
        #         for m in st.session_state.messages
        #     ],
        #     stream=True,
        # )
        
        # grab user/system message from session state
        user_message = ChatMessage.from_system(st.session_state["user_message"])
        system_message = ChatMessage.from_system(st.session_state["system_message"])
        messages = [user_message,  system_message]
        print(st.session_state.messages)
        print(st.session_state.messages[-1]["content"])
        print("   🔎 haystack memory: ",memory_store.messages)
        print("content 0" ,st.session_state.messages[0]["content"])
        print("content -1" ,st.session_state.messages[-1]["content"])
        # res = pipeline.run(data={"retriever": {"query": st.session_state.messages[0]["content"]},
        #                         "prompt_builder": {"template": messages, "query": st.session_state.messages[0]["content"]},
        #                         "memory_joiner": {"value": [ChatMessage.from_user(st.session_state.messages[0]["content"])]}},
        #                         include_outputs_from=["llm"])
        # res = conversational_rag.run(data={"query_rephrase_prompt_builder": {"query": st.session_state.messages[0]["content"]},
        #                         "prompt_builder": {"template": messages, "query": st.session_state.messages[0]["content"]},
        #                         "memory_joiner": {"value": [ChatMessage.from_user(st.session_state.messages[0]["content"])]}},
        #                         include_outputs_from=["llm","query_rephrase_llm"])
        res = conversational_rag.run(data={"query_rephrase_prompt_builder": {"query": st.session_state.messages[-1]["content"]},
                             "prompt_builder": {"template": messages, "query": st.session_state.messages[-1]["content"]},
                             "memory_joiner": {"value": [ChatMessage.from_user(st.session_state.messages[-1]["content"])]}},
                            include_outputs_from=["llm"])
        
        stream = res['llm']['replies'][0]
        print("items: ",res.items())
        search_query = res['query_rephrase_llm']['replies'][0]
        print(f"   🔎 Search Query: {search_query}")
        def generate_stream(text):
            for word in text.split(" "):
                yield word + " "
        print(stream.content)
        # print("Prompt Builder:",res["prompt_builder"])
        print("ALL:", res["retriever"])
        response = st.write_stream(generate_stream(stream.content))
    st.session_state.messages.append({"role": "assistant", "content": response})