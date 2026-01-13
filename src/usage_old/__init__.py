import os
import sys
import chromadb
import textwrap

from openai import OpenAI
from chromadb.utils import embedding_functions
import textwrap


def get_openai_embeddings(text, client):
    response = client.embeddings.create(input=[text], model="text-embedding-3-small")

    embedding = response.data[0].embedding
    print("Generating embedding")

    return embedding


# function to query documents
def query_documents(question, client, collection):
    query_embeddings = get_openai_embeddings(question, client)

    results = collection.query(query_embeddings=query_embeddings)

    # Extract relevant chuncks
    relevant_chuncks = [doc for sublist in results["documents"] for doc in sublist]

    print("getting relevant information")
    # print(results)

    return relevant_chuncks


# function to generate a response from OpenAI

def generete_response(question, relevant_chuncks, client):
    context = "\n\n".join(relevant_chuncks)

    # prompt = (
    # "You are scientific assistant tasked with answering questions. Below we provide some un-formated context that might or might not be relevant to the asked question. If it is relevant, be sure to use it to deliver concrete and concise answers. Give precise details. Don't use overly flowery voice." +
    # "### QUESTION\n" + question + "\n\n"
    # "### CONTEXT\n" + context
    # )
    prompt = (f"""
You are a highly knowledgeable and precise scientific assistant, designed to assist researchers, scientists, 
and professionals by answering questions based on retrieved scientific literature. You process, summarize and synthesize 
information from relevant database chunks while maintaining clarity, conciseness, and scientific accuracy.

### Important Considerations:
- **Not all retrieved chunks will be relevant.** Some may contain unrelated, incorrect, or misleading information.
- **Your task is to critically evaluate the chunks, extract only what is relevant, and discard anything irrelevant or misleading.**
- **Do not assume all retrieved information is applicable.** Verify coherence with known scientific principles and the user's question.

### Guidelines for Answering:

1. **Prioritize Relevance:**
   - Analyze the retrieved chunks and extract only the information directly relevant to the user's question.
   - Ignore unrelated details, speculative claims, or low-quality information.

2. **Ensure Scientific Rigor:**
   - Base responses on evidence from the retrieved sources while maintaining logical consistency.
   - If multiple interpretations exist, present them objectively and indicate their level of support.

3. **Summarize, Don't Just Relay:**
   - Rephrase complex findings for clarity while preserving technical accuracy.
   - If necessary, cite key findings concisely rather than quoting verbatim.
   - Avoid blindly trusting any single chunk; cross-check against multiple retrieved chunks if available.

4. **Handle Uncertainty Transparently:**
   - If the retrieved data does not fully answer the question, acknowledge the gap.
   - Suggest possible interpretations or areas for further research rather than making unsupported claims.

5. **Concise and Structured Responses:**
   - Provide a direct answer first, followed by supporting details.
   - Use bullet points or structured explanations when appropriate.

6. **Avoid Speculation and Noise:**
   - Do not generate conclusions beyond what the retrieved data supports.
   - Clearly distinguish between well-supported findings and inconclusive or weak evidence.
   - If external knowledge is needed, state that explicitly instead of making assumptions.

Your goal is to provide scientifically sound, relevant, and concise responses, filtering out noise and misleading information while ensuring the highest degree of accuracy.

### BEGINNING OF CHUNKS
{context}              
""")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question
            },
        ],
    )

    answer = response.choices[0].message

    return answer


def chat_gpt(client, collection):
    # Example query
    # query documents

    # example question and response generation
    # question = "What is the biological function of hGID"

    question = input("Digit your question here: ")

    relevant_chunkcs = query_documents(question.strip(), client, collection)

    answer = generete_response(question.strip(), relevant_chunkcs, client)

    print("############################################################")
    print(textwrap.fill(answer.content, width=100))
    print("############################################################")


if __name__ == "__main__":

    # get openai key

    openai_key = os.getenv("OPENAI_API_KEY")

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai_key, model_name='text-embedding-3-small')

    # initialize chroma client
    # to fix, chroma db doesn' recognise absolut path
    # path = "C:/Users/dario/OneDrive/università/MA/Thesis/microscope-toolset/microscope-toolset/chroma_storage"
    path = "/chroma_storage"
    chroma_client = chromadb.PersistentClient(path="./chroma_storage")
    collection_name = "semantic_much_bigger_qa_collection"
    collection = chroma_client.get_collection(name=collection_name, embedding_function=openai_ef)

    client = OpenAI(api_key=openai_key)

    choice = ""

    while choice != "quit":

        choice = input(
            "Available command:\n Press 'quit' to terminate\n Press 'chat' to start conversation with GPT-4\n\n : ")

        command = choice.lower().strip()

        if command == "quit":
            sys.exit("See you next time!")

        elif command == "chat":
            chat_gpt(client, collection)

        else:
            print("Please choose a valid option ('quit', 'chat')")



