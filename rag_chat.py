import streamlit as st

# LLM Model

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

# PDF preprocess
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embedding Model
from langchain.embeddings.openai import OpenAIEmbeddings


# Memory buffer
from langchain.memory import ConversationBufferMemory
from langchain.memory import StreamlitChatMessageHistory

# Vector DB

from langchain.vectorstores import FAISS


from langchain.callbacks import get_openai_callback

import yaml
import numpy as np
from loguru import logger

def main():
    st.set_page_config(
        page_title = 'Hist rag chat make_kh',
        page_icon = ':flag-kh:'
    )
    st.title(":blue[Hist] _Chat_ :robot_face:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    # if 'checker' not in st.session_state:
    #     st.session_state.checker = None
    # if 'emb' not in st.session_state:
    #     st.session_state.emb = None
    with st.sidebar:
        uploaded_files = st.file_uploader("File을 업로드 하세요",type=['pdf'],accept_multiple_files=True)
        openai_api_key = st.text_input('OpenAI API Key', key = 'chatbot_api_key', type= 'password')
        process = st.button("실행")

    # start
    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks,openai_api_key)

        st.session_state.conversation = conversation_chat(vetorestore, openai_api_key)

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role" : "assistant",
                                         "content" : "안녕하세요 궁금한 점이 있으신가요?"}]

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    history = StreamlitChatMessageHistory(key='chat_messages')

    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content" : query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("답을 찾는 중입니다..."):
                result = chain({"question" : query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                print(st.session_state.chat_history)
                response = result['answer']
                source_documents = result['source_documents']

                print(source_documents)
                st.markdown(response)
                # with st.expander("참고 문서 확인"):
                    # st.markdown(source_documents[0].metadata['source'], help=source_documents[0].page_content)
                    # st.markdown(source_documents[1].metadata['source'], help=source_documents[1].page_content)
                    # st.markdown(source_documents[2].metadata['source'], help=source_documents[2].page_content)
        st.session_state.messages.append({'role' : "assistant", 'content' : response})

def get_text(docs):
    doc_list = []

    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks,api_key):
    embed_model = OpenAIEmbeddings(model='text-embedding-ada-002',openai_api_key = api_key)
    vectordb = FAISS.from_documents(text_chunks, embed_model)
    return vectordb

def conversation_chat(vectordb,api_key):
    llm = ChatOpenAI(
        openai_api_key = api_key,
        model = 'gpt-3.5-turbo',
        temperature = 0
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        # combine_docs_chain_kwargs={"prompt": template},
        chain_type = 'stuff',
        retriever = vectordb.as_retriever(vervose=True, k=3),
        # retriever=vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.01,
        #                                                                                          'k': 3}),
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'), # anwser는 답변만 답겠다.
        get_chat_history = lambda h:h,
        return_source_documents = True,
        verbose = True
    )
    return conversation_chain
if __name__ == '__main__':
    main()
