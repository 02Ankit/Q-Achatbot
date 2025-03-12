# This simple takes each chunk of document, passes it to the LLm along side the prompt question
# An answer to that prompt question is generated. From here the next chunk or 
# document is passes to the LLM and the first answer is refine or fine tuned based on the information from this document
# this is repeated for all the othr documents till a correct answer is obtained.
# Number of calls to the LLM is proportionate to the number of documents

# import langchain
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from decouple import config
import os
import streamlit as st
# from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings


TEXT = ["Python is a versatile and widely used programming language known for its clean and readable syntax, which relies on indentation for code structure",
        "It is a general-purpose language suitable for web development, data analysis, AI, machine learning, and automation. Python offers an extensive standard library with modules covering a broad range of tasks, making it efficient for developers.",
        "It is cross-platform, running on Windows, macOS, Linux, and more, allowing for broad application compatibility."
        "Python has a large and active community that develops libraries, provides documentation, and offers support to newcomers.",
        "It has particularly gained popularity in data science and machine learning due to its ease of use and the availability of powerful libraries and frameworks."]

meta_data = [{"source": "document 1", "page": 1},
             {"source": "document 2", "page": 2},
             {"source": "document 3", "page": 3},
             {"source": "document 4", "page": 4}]

# embedding_function = OpenAIEmbeddings(
#     model = "text-embedding-3-small"
# )

embedding_function = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2") 


vector_db = Chroma.from_texts(
texts = TEXT,
persist_directory = "../vector_db1",
collection_name = "HR_resources_system",
embedding = embedding_function
    
)
    


# QA_prompt = PromptTemplate(
#     template = """Use the following pieces of information to answer the user
#     Context: {context}
#     Questions: {question}
#     Answer:""",
#     input_variables = ["text", "question"]
# )




llm = ChatOpenAI(openai_api_key=st.secrets("OPENAI_API_KEY"), temperature = 0.6)

memory = ConversationBufferMemory(
    return_messages = True, 
    memory_key= "chat_history",
    output_key="answer" 
)


qa_chain = ConversationalRetrievalChain.from_llm(
    llm = llm,
    memory = memory,
    retriever = vector_db.as_retriever(search_kwargs = {"fetch_k": 4, "k": 3}, search_type = "mmr"),
    return_source_documents = True,
    chain_type = 'refine'
    
)

def rag_func(question: str) -> str:
    """
    This function takes in user question or prompt and return as response.
    :param: question: String value of the question or the prompt from the user. 
    :returns: String value of the answer to the user question.
    
    """
    response = qa_chain.invoke({"question": question})

    return response.get("answer")

    # resp = rag_func()
    # print(resp)







# print("============================================")
# print("===============Source Documents============")
# print("============================================")

# print(response["source_documents"][0])
