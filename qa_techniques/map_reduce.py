# In Map Reduce, each document or chunk is sent to the LLM to obtain an original answer, 
# these original answer every documents send to LLm and combines are then composed into one final answer
# this involves more LLm calls

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
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
    texts=TEXT,
    embedding=embedding_function,
    metadatas=meta_data
)

QA_prompt = PromptTemplate(
    template = """Use the following pieces of context to answer the question of user
    Context: {context}
    Questions: {question}
    Answer:""",
    input_variables = ["text", "question"]
)

combine_prompt = PromptTemplate.from_template(
    template = """Write a summary of the folowing text \n\n{summaries}"""
)

question_prompt = PromptTemplate.from_template(
    template="""Use the following piece of context to answer the question
    context: {context}}
    question: {question}
    HelpFul answer:"""
)


llm = ChatOpenAI(openai_api_key = config("OPENAI_API_KEY"), temperature = 0.3)

qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    retriever = vector_db.as_retriever(search_kwargs = {"fetch_k": 4, "k": 3}, search_type = "mmr"),
    return_source_documents = True,
    chain_type = 'map_reduce',
    chain_type_kwargs = {"question_prompt": question_prompt, "combine_prompt": combine_prompt} 
)

question = "what area is python mostly used"

response = qa_chain({"query": question})


print("============================================")
print("====================Result==================")
print("============================================")
print(response["result"])


print("============================================")
print("===============Source Documents============")
print("============================================")

print(response["source_documents"][0])