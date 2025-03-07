# This simple takes each chunk of document, passes it to the LLm along side the prompt question
# An answer to that prompt question is generated. From here the next chunk or 
# document is passes to the LLM and the first answer is refine or fine tuned based on the information from this document
# this is repeated for all the othr documents till a correct answer is obtained.
# Number of calls to the LLM is proportionate to the number of documents

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from decouple import config #open .env file

TEXT = ["Python is a versatile and widely used programming language known for its clean and readable syntax, which relies on indentation for code structure",
        "It is a general-purpose language suitable for web development, data analysis, AI, machine learning, and automation. Python offers an extensive standard library with modules covering a broad range of tasks, making it efficient for developers.",
        "It is cross-platform, running on Windows, macOS, Linux, and more, allowing for broad application compatibility."
        "Python has a large and active community that develops libraries, provides documentation, and offers support to newcomers.",
        "It has particularly gained popularity in data science and machine learning due to its ease of use and the availability of powerful libraries and frameworks."]

meta_data = [{"source": "document 1", "page": 1},
             {"source": "document 2", "page": 2},
             {"source": "document 3", "page": 3},
             {"source": "document 4", "page": 4}]

embedding_function = HuggingFaceEmbeddings(
    model_name = "all-MiniLM-L6-V2"
)

vector_db = Chroma.from_texts(
    texts = TEXT,
    persist_directory = "../vector_db",
    embedding = embedding_function,
    collection_name = "HR_resources_system" 
)

# QA_prompt = PromptTemplate(
#     template = """Use the following pieces of information to answer the user
#     Context: {context}
#     Questions: {question}
#     Answer:""",
#     input_variables = ["text", "question"]
# )

llm = ChatOpenAI(openai_api_key = config("OPENAI_API_KEY"), temperature = 0.6)

memory = ConversationBufferMemory(
    return_messages = True, memory_key= "chat_history" 
)


qa_chain = ConversationalRetrievalChain.from_llm(
    llm = llm,
    memory = memory,
    retriever = vector_db.as_retriever(search_kwargs = {"fetch_k": 4, "k": 3}, search_type = "mmr"),
    return_source_documents = True,
    chain_type = 'refine',
     
)

question = "what is the HUMAN RESOURCES"

response = qa_chain({"query": question})


print("============================================")
print("====================Result==================")
print("============================================")
print(response.get("answer"))


# print("============================================")
# print("===============Source Documents============")
# print("============================================")

# print(response["source_documents"][0])