import os

from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
#import pyhtml2md


class ContextualizationOutput(BaseModel):

    intent: str
    message: str

def download_html(url: str, file_path: str = None) -> None:
    """
    From a web page, download the html content and save it into a file
    """
    response = requests.get(url)

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # write html file into the dir
    if file_path is None:
        file_path = os.getcwd().replace("\\", "/")
    else:
        file_path = os.path.abspath(file_path).replace("\\", "/")

    # write the files
    file_name = os.path.basename(url)
    with open(f"{file_path}/{file_name}.html", "w", encoding="utf-8") as file:

        file.write(str(soup))

    # close file
    file.close()

def contextualize_chunks(client, document, chunk):
    """
    This function given a whole doc and a chunk try to contextualize the
    chunk in a meaningful way.
    """
    PROMPT = """
    Whole document:
    {doc_content}
    
    Here is the chunk we want to situate within the whole document:
    {chunk_content}
    """
    user_query = """
    Please give a short succinct context to situate this chunk within the overall document context for the purposes of improving search retrieval of the chunk.
    Answer only with the succinct context and nothing else.
    """
    prompt = PROMPT.format(doc_content=document, chunk_content=chunk)

    response = client.responses.parse(
        model="gpt-4.1-mini",
        input=[{"role": "system", "content": prompt}, {"role": "user", "content": user_query}],
        text_format=ContextualizationOutput
    )

    parsed_response = response.output_text

    # The output should remain a json object that we will save inside the databases

    return parsed_response

def parse_to_markdown(file_path: str = None) -> None:
    """
    Transform html file into a markdown
    """

    with open(file_path, "r", encoding='utf-8') as file:

        text = file.read()

    file.close()

    markdown = pyhtml2md.convert(text)
    # write markdown
    file_name = os.path.basename(file_path)
    file_dir = os.path.dirname(file_path)
    t = open(f"{file_dir}/{file_name}_test_docs.md", "w")

    t.write(markdown)

    t.close()

    print("Document parsed successfully.")
