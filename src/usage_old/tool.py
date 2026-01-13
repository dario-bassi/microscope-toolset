import os
from typing import List
import pymupdf4llm
import pymupdf
import pathlib
from openai import OpenAI

from semantic_text_splitter import MarkdownSplitter



def convert_pdf_to_markdown(path_to_directory: str) -> List[dict]:
    """
    It converts to pdf file text into markdown

    Input -> path/to/directory/pdfs
    Output -> List[dictonary(metadata, text)]
    """

    print("---------Transforming documents into markdown file from directory-------------")

    if not os.path.exists(path_to_directory):
        os.makedirs(path_to_directory)
    documents = []
    for filename in os.listdir(path_to_directory):
        if filename.endswith(".pdf"):
            print(os.path.join(path_to_directory,filename))
            # create path of the file
            input_path = os.path.join(path_to_directory,filename)
            # create output of the markdown file (optional)
            out_path = os.path.join(path_to_directory,f'{filename.split(".")[0]}.md')
            # extract metadata
            load_file = pymupdf.load(input_path)
            # convert the text into markdown type
            md_text = pymupdf4llm.to_markdown(input_path)
            # write md file (optional)
            pathlib.Path(out_path).write_bytes(md_text.encode())
            # save text into a list
            documents.append({"id": filename.split(".")[0], "title": load_file.metadata.title,"text": md_text})
            
    
    return documents

def split_markdown(text: str, max_characters: int) -> List[str]:
    """
    This function split the markdown text into a list of semantic chunk
    """

    # Optionally can also have the splitter not trim whitespace for you
    splitter = MarkdownSplitter(max_characters)
    # splitter = MarkdownSplitter(max_characters, trim=False)

    chunks = splitter.chunks(text)

    return chunks


def split_text(document: List[dict], chunk_size = 5000) -> List[dict]:

    """
    This function split the markdown text into different chunks of the size of value chunk_size
    """
    # chunk size of @chunk_size charachters
    chuncked_documents = []

    for elem in document:

        splitted_text_list = split_markdown(elem["text"], chunk_size)

        for i, chunk in enumerate(splitted_text_list):
             chuncked_documents.append({"id":f'{elem["id"]}',"chunk": f'{i+1}', "title": elem["title"],"text": chunk})

    
    return chuncked_documents


def get_openai_embeddings(text: str, client: OpenAI) -> List[str]:
    """
    This function return the embedding vector calculated by OpenAI
    """

    response = client.embeddings.create(input=[text], model="text-embedding-3-small")

    embedding = response.data[0].embedding
    print("Generating embedding")

    return embedding