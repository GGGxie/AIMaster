from langchain.chains import RetrievalQA,create_retrieval_chain,create_history_aware_retriever
from langchain_core.messages import HumanMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatZhipuAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from vecdb.vectordb import get_vectordb
store={}
class Chat:
    def __init__(self,
                 llm,
        ):
        self.llm = llm
        self.store = {}
        ### Contextualize question ###
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        vectordbt = get_vectordb()
        retriever=vectordbt.as_retriever()
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        ### Answer question ###
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        self.conversational_rag_chain = RunnableWithMessageHistory(
            runnable=rag_chain,
            get_session_history=self.__get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )


    def __get_session_history(self,session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]

    def run(
            self,
            task: str,
            sessionid="1",
    ) -> str:
        return self.conversational_rag_chain.invoke(
            {"input": task},
            config={
                 "configurable": {"session_id":sessionid}
                }
    )["answer"]