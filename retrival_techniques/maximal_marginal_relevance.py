
#maximal marginal relevance (MMD)
# this does not only give you the right context to work with, but also ensure diversity.

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma

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

response = vector_db2.max_marginal_relevance_search(
    query = "tell me about a programming langauge which is use to datascience", 
    k=2,
    fetch_k= 3
)

print("response:", response)


#compress relevant document into only two or three sentences containing the exact information
# you need with the help of an LLM. from here we can then make a final call to the llm passing 
# in the two or three sentences that contain the exact information needed to answer the question
# down side of this is the cosr of making so many llm calls needed to compress down the information
# we retrived.
#  