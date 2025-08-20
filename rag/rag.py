from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv
load_dotenv()

# loading text
from langchain_community.document_loaders import PyPDFLoader


pdf_url = "https://raw.githubusercontent.com/Sajidcodes/LangGraph/main/rag/DMLS.pdf"
# returns a document object
loader = PyPDFLoader(pdf_url)

docs = loader.load()

print(len(docs))

from pprint import pprint

import json
# text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=2500,
    chunk_overlap=300,
)

# Perform the split
# pass the text not the document
# full book chunking

all_chunks= []
for doc in docs:
    chunks = splitter.split_text(doc.page_content) 
    all_chunks.extend(chunks)

# print(len(chunks))
# print(chunks[0])
print(all_chunks)
print(f"Total chunks: {len(all_chunks)}")


# ### Embed chunks

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()


# ### Store in Vector Store

# store chunks + embeddings in chroma db
vectorstore = Chroma.from_texts(
    texts=all_chunks,       # the list of chunks you created
    embedding=embeddings,   # your embedding model
    persist_directory="chroma_db",  # folder name for saving
    collection_metadata={"hnsw:space": "cosine"} # l2,ip

)

# persist to disk
vectorstore.persist()
print("✅ Stored chunks into Chroma DB")

# # reload the db w/o building
# vectorstore = Chroma(
#     persist_directory="rag_chroma",
#     embedding_function=embeddings
# )
# print(vectorstore._collection.count())  # number of stored chunks


query = 'creating production grade ML systems?'


# ## Augmentation


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-4.1-mini')

# 0. creating the retriever from your chroma vs
retriever = vectorstore.as_retriever(search_kwargs={'k':2})

query = 'production best practices'


# get the raw retrieve documents, shouldn't be used directly in the pipeline
retrieved_docs = retriever.get_relevant_documents(query)

#prepare context
context = "\n\n".join([d.page_content for d in retrieved_docs])

# print content = exactly what is sent to context
for i, doc in enumerate(retrieved_docs):
    print(f"--- Document {i+1} ---")
    print(doc.page_content)


from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# 1. defining prompt style
# prompt arranges retrieved info into context
prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant.
    Answer ONLY from the provided document context.
    If the context is insufficient, just say 
    answer not found in the database
    
    Context:
    {context}

    Question: {question}
    """
)

# 2. defining a parser
parser = StrOutputParser()


# 3. Build the pipeline

from langchain.schema.runnable import RunnablePassthrough
# Runnablepassthrough() passes the question through the pipeline

rag_pipeline = ({
    "context": retriever,
    "question": RunnablePassthrough()} | prompt | llm | parser
)

