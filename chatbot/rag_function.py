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
# from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings


TEXT = ["Employees are eligible for 10 days of sick leave, 15 days of paid leave, and 10 public holidays per year. Leaves must be applied for in advance via the HR portal, except in emergencies.",
        "Employees are paid on the last working day of each month. Payslips are available for download from the HR portal. For discrepancies, contact payroll@company.com.",
        "New employees undergo a probation period of 6 months. Performance is reviewed before confirmation. Employees are notified of their confirmation status via email.",
        "Employees can apply for WFH up to 5 days per month. Approval depends on the manager and HR. Emergency WFH can be requested directly via email.",
        "The company provides medical insurance coverage for employees and their dependents. Coverage includes hospitalization and maternity. Insurance cards are issued within 30 days of joining.",
        "Employees can report grievances confidentially to hr.support@company.com or use the HR portal. All grievances are addressed within 7 working days.",
        "Employees must serve a notice period of 30 days. All company assets must be returned before the last working day. Full and final settlement is processed within 45 days of exit.",
        "Performance appraisals are conducted annually in March. Managers and employees jointly review performance and set goals for the next year. Promotions and raises depend on performance ratings.",
        "The company offers online and offline training programs. Employees can register via the HR portal. Trainings aim at skill development and career growth.",
        "All employees are expected to adhere to the company’s Code of Conduct. Policies on harassment, discrimination, and workplace behavior are strictly enforced. Violations may lead to disciplinary action."]

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

import streamlit as st



llm = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], temperature = 0.6)

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


def get_context(user_question):
    mapping = {
        "leave": "Employees are eligible for 10 days of sick leave, 15 days of paid leave, and 10 public holidays per year. Leaves must be applied for in advance via the HR portal, except in emergencies.",
        "salary": "Employees are paid on the last working day of each month. Payslips are available for download from the HR portal. For discrepancies, contact payroll@company.com.",
        "probation": "New employees undergo a probation period of 6 months. Performance is reviewed before confirmation. Employees are notified of their confirmation status via email.",
        "wfh": "Employees can apply for WFH up to 5 days per month. Approval depends on the manager and HR. Emergency WFH can be requested directly via email.",
        "insurance": "The company provides medical insurance coverage for employees and their dependents. Coverage includes hospitalization and maternity. Insurance cards are issued within 30 days of joining.",
        "grievance": "Employees can report grievances confidentially to hr.support@company.com or use the HR portal. All grievances are addressed within 7 working days.",
        "resignation": "Employees must serve a notice period of 30 days. All company assets must be returned before the last working day. Full and final settlement is processed within 45 days of exit.",
        "appraisal": "Performance appraisals are conducted annually in March. Managers and employees jointly review performance and set goals for the next year. Promotions and raises depend on performance ratings.",
        "training": "The company offers online and offline training programs. Employees can register via the HR portal. Trainings aim at skill development and career growth.",
        "conduct": "All employees are expected to adhere to the company’s Code of Conduct. Policies on harassment, discrimination, and workplace behavior are strictly enforced. Violations may lead to disciplinary action."
    }

    for keyword in mapping:
        if keyword in user_question.lower():
            return mapping[keyword]
    return "I'm sorry, I couldn't find an appropriate policy for your question. Please contact HR for more information."


def rag_func(question: str) -> str:

    context = get_context(question)

    QA_prompt = PromptTemplate(
        template = """You are an HR assistant chatbot designed to answer employees' 
        HR-related queries in a clear, polite, and professional manner. 
        Please answer concisely and accurately.
        Context: {context}
        Questions: {question}
        Answer:""",
        input_variables = ["context", "question"]
    )

    # """
    # This function takes in user question or prompt and return as response.
    # :param: question: String value of the question or the prompt from the user. 
    # :returns: String value of the answer to the user question.
    
    # """
    formatted_prompt = QA_prompt.format(context = context, question=question)
    response = qa_chain.invoke({"question": formatted_prompt})

    # response = qa_chain.invoke({"question": formatted_prompt})

    return response.get("answer", "sorry i couldn't find an answer to that.")

    # resp = rag_func()
    # print(resp)







# print("============================================")
# print("===============Source Documents============")
# print("============================================")

# print(response["source_documents"][0])
