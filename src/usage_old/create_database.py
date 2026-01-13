import os
import chromadb
import pathlib
import pymupdf4llm

from openai import OpenAI
from chromadb.utils import embedding_functions
from semantic_text_splitter import MarkdownSplitter
import tool


def convert_pdf_to_markdown(path_to_directory):

    print("---------Transforming documents into markdown file from directory-------------")

    if not os.path.exists(path_to_directory):
        os.makedirs(path_to_directory)
    documents = []
    for filename in os.listdir(path_to_directory):
        if filename.endswith(".pdf"):
            print(os.path.join(path_to_directory,filename))
            input_path = os.path.join(path_to_directory,filename)
            out_path = os.path.join(path_to_directory,f'{filename.split(".")[0]}.md')
            md_text = pymupdf4llm.to_markdown(input_path)
            # write md file
            pathlib.Path(out_path).write_bytes(md_text.encode())
            # save text into a list
            documents.append({"id": filename.split(".")[0], "text": md_text})
            
    
    return documents



def split_markdown(text):

    # Maximum number of characters in a chunk
    max_characters = 1000
    # Optionally can also have the splitter not trim whitespace for you
    splitter = MarkdownSplitter(max_characters)
    # splitter = MarkdownSplitter(max_characters, trim=False)

    chunks = splitter.chunks(text)

    return chunks

def main():

    # verify that the user has an open ai key
    if os.getenv("OPENAI_API_KEY") is None:
        print("You must add an openai key as local variable")
        # later create error
    else:
         openai_key = os.getenv("OPENAI_API_KEY")

    # create embendings function used by the vector database
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai_key,model_name='text-embedding-3-small')

    # initialize chroma client
    chroma_client = chromadb.PersistentClient(path="../chroma_storage")
    # create collection name
    collection_name = "vector_database"
    # initialize a new collection or an existing one
    collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=openai_ef)

    client = OpenAI(api_key=openai_key)

    # later add argparse

    # load nad convert pdfs file
    documents = tool.convert_pdf_to_markdown("./publications")

    # split the text into chuks (later add value given by user)
    chunked_docs = tool.split_text(documents)

    
    for doc in chunked_docs:
        # calculate the embedding vector
        doc["embedding"] = tool.get_openai_embeddings(doc["text"], client)
        # Insert the metadata, chunked docs and embedding vector into the database
        print("--------Inserting chuncks into db---------")
        collection.upsert(ids=[doc["id"]], metadatas=[{"chunk": f'{doc["chunk"]}',"title": doc["title"]}],  documents=[doc["text"]], embeddings=[doc["embedding"]])
    
    print("Database was created and is ready to use!!")

    

if __name__ == "__main__":

    main()