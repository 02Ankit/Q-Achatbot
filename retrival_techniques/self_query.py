#This is used when the question asked or the text we want to search 
# the vector store against is just not to deal with semantic but 
# also meta-data about the semantics
#  
import langchain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from decouple import config #open .env file

TEXT = ["python is a versatile and widley used programming language",
        "it is a general-purpose language suitable for web development, data analytics, AI machine learning, deep learning and generative ai ",
        "it is cross-plateform, running on windows, macOS, linux, ubuntu",
        "python has a large and active community that develops multi linguality language",
        "it has particularly gained popularity in data science and machine learning, due to its easy of use and the availability of powerful libraries and framework."]

meta_data = [{"source": "document 1", "pages":1},
             {"source": "document 2", "pages":2},
             {"source": "document 3", "pages":3},
             {"source": "document 4", "pages":4}]

embedding_function = HuggingFaceEmbeddings(
    model_name = "all-miniLM-L6-V2"
)

vector_db2 = Chroma.from_texts(
    texts = TEXT,
    embedding = embedding_function,
    metadatas = meta_data
)

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="This is the source documents there are 4 main documents,  `document 1`, `document 2`, `document 3`, `document 4`",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the details of Python",
        type="integer",
    ),
]

document_content_description = "Info on Python Programming Language"
llm = OpenAI(temperature=0, openai_api_key=config("OPENAI_API_KEY"))

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vector_db2,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    verbose=True
)


docs = retriever.get_relevant_documents(
    "What was mentioned in the 4th document about  Python")
print(docs)