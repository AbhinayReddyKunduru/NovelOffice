from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from constants import *

import os
os.environ["OPENAI_API_KEY"] = secret_key

embeddings = OpenAIEmbeddings()

docsearch = Chroma(persist_directory="db", embedding_function=embeddings)

chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="stuff", retriever=docsearch.as_retriever())

user_input = input("What's your question: ")

result = chain({"question": user_input}, return_only_outputs=True)

print(result)

print("Answer: " + result["answer"].replace('\n', ' '))
print("Source: " + result["sources"])
