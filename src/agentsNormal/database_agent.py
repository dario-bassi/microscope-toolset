import logging

from typing import List

from cmap.data.cmasher import torch
from openai import OpenAI

from src.agentsNormal.structuredOutput import RephraseOutput, ExtractKeywordOutput
# from contextualization_pymmcore import parsed_response
# from prompts.databaseAgentPrompt import DATABASE_PROMPT
from src.postqrl.log_db import LoggerDB
from src.databases.elasticsearch_db import ElasticSearchDB
import json
from typing import Any
import torch


class DatabaseAgent:

    # def __init__(self, client_openai: OpenAI, db_log: LoggerDB, db_log_collection_name: str,
    #              chroma_client: chromadb.ClientAPI = None,
    #              client_collection: chromadb.Collection = None):
    #     self.client_openai = client_openai
    #     # self.chroma_client = chroma_client
    #     # self.client_collection = client_collection
    #     self.db_log = db_log
    #     self.db_log_name = db_log_collection_name

    def __init__(self, client_openai: OpenAI, es_client: ElasticSearchDB, pdf_collection: str,
                 micromanager_collection: str, api_collection: str, db_log: LoggerDB, db_log_collection_name: str,
                 tokenizer: Any, model: Any):
        self.client_openai = client_openai
        # self.chroma_client = chroma_client
        # self.client_collection = client_collection
        self.es_client = es_client
        self.pdf_collection = pdf_collection
        self.micromanager_collection = micromanager_collection
        self.api_collection = api_collection
        self.db_log = db_log
        self.db_log_name = db_log_collection_name
        self.tokenizer = tokenizer
        self.model = model

    def embeds_query(self, query) -> List[float]:

        
        response = self.client_openai.embeddings.create(input=[query], model="text-embedding-3-small",
                                                        dimensions=512)  # later add model's choice

        embedding = response.data[0].embedding  # list of floating values
        print("Generating embedding")

        return embedding

    def retrieve_relevant_information(self, query: str):

        try:
            # embed the user query
            query_embedded = self.embeds_query(query=query)

            # search into the database for chroma
            # results = self.client_collection.query(query_embeddings=query_embedded, n_results=25)
            # search into the db of Elasticsearch
            # 1. search into the api collection
            api_docs_result = self.es_client.hybrid_search(index_name=self.api_collection, query=query,
                                                           query_vector=query_embedded,
                                                           keyword_to_search_for_bm25="contextualize_text")
            # 2. search into pdf collection
            pdf_docs_results = self.es_client.hybrid_search(index_name=self.pdf_collection, query=query,
                                                            query_vector=query_embedded,
                                                            keyword_to_search_for_bm25="content")
            # 3. search into micromanager devices
            micromanager_docs_results = self.es_client.hybrid_search(index_name=self.micromanager_collection, query=query,
                                                                     query_vector=query_embedded,
                                                                     keyword_to_search_for_bm25="content")
            # now extract list of object for reranking
            list_api_docs_result = [{
                "type": result["_source"]["type"],
                "name": result["_source"]["name"],
                "signature": result["_source"]["signature"],
                "description": result["_source"]["description"],
                "filename": result["_source"]["filename"],
                "contextualize_text": result["_source"]["contextualize_text"],
                "score": result["_score"]
            }
                for result in api_docs_result["hits"]["hits"]
            ]
            list_pdf_docs_results = [{
                "content": result["_source"]["content"],
                "chunk_id": result["_source"]["chunk_id"],
                "filename": result["_source"]["filename"],
                "score": result["_score"]
            }
                for result in pdf_docs_results["hits"]["hits"]
            ]
            list_micromanager_docs_results = [{
                "doc_id": result["_source"]["doc_id"],
                "content": result["_source"]["content"],
                "chunk_id": result["_source"]["chunk_id"],
                "filename": result["_source"]["filename"],
                "score": result["_score"]
            }
                for result in micromanager_docs_results["hits"]["hits"]
            ]
            # sort list of results
            list_api_docs_result = sorted(list_api_docs_result, key=lambda x: x["score"], reverse=True)
            list_micromanager_docs_results = sorted(list_micromanager_docs_results, key=lambda x: x["score"], reverse=True)
            list_pdf_docs_results = sorted(list_pdf_docs_results, key=lambda x: x["score"], reverse=True)

            # merge all result into a single list
            merged_list = []
            # iterate through the first list
            merged_list = self.rerank_and_add_to_list(merged_list=merged_list, partial_result_list=list_api_docs_result,
                                                      keyword_field="contextualize_text", query=query)
            # iterate through the second list
            merged_list = self.rerank_and_add_to_list(merged_list=merged_list, partial_result_list=list_pdf_docs_results,
                                                      keyword_field="content", query=query)
            # iterate through the last list
            merged_list = self.rerank_and_add_to_list(merged_list=merged_list,
                                                      partial_result_list=list_micromanager_docs_results,
                                                      keyword_field="content", query=query)
            #print(merged_list)
            # sort the list descending, from higher score through lower
            sorted_merged_list = sorted(merged_list, key=lambda x: x["score"], reverse=True)
            #print(sorted_merged_list)

            # Just keep first 25 results
            results = sorted_merged_list[:24]
            #print(results)

            # search into the database for the log
            log_result = self.db_log.query_by_vector(collection_name=self.db_log_name, vector=query_embedded, k=5)
            # print(log_result)
            if log_result is None:
                log_result = []

            relevant_information = [json.dumps(chunk) for chunk in results]

            # log_relevant_chunks
            log_chunks = [
                json.dumps({
                    "prompt": log_result[i]['prompt'],
                    "output": log_result[i]['output'],
                    "feedback": log_result[i]['feedback'],
                    "category": log_result[i]['category']
                })
                for i in range(len(log_result))
            ]

            # Extract relevant chuncks
            SIMILARITY_THREASHOLD = 0.80

            # result

            # relevant_chuncks = [
            #     f"\n **Function:** {results['metadatas'][0][i]['function_name']}\n **Signature:** {results['metadatas'][0][i]['signature']}\n **Description:** {results['metadatas'][0][i]['description']}\n **Doc Snippet:**\n{results['documents'][0][i]}"
            #     for i in range(len(results["documents"][0]))]
            # relevant_chuncks = [
            #     results['documents'][0][i] for i in range(len(results["documents"][0]))
            # ]
            # relevant_chuncks = [doc for sublist in results["documents"] for doc in sublist]
            # relevant_chuncks = []
            # for index, doc in enumerate(results["documents"][0]):
            #    # transform distances into a similsrity score
            #    score = self.similarity_score(results["distances"][0][index])
            #    print(score)
            #
            #    if score >= SIMILARITY_THREASHOLD:
            #        relevant_chuncks.append(doc)

            # print(log_chunks)
            print("getting relevant information")

            joined_chunks = [*relevant_information, *log_chunks]

            return joined_chunks
        except Exception as e:
            logging.error(e)

    def look_for_context(self, query: str) -> str:

        # retrieve relvant informations
        list_of_relevant_informations = self.retrieve_relevant_information(query=query)

        if len(list_of_relevant_informations) == 0 or list_of_relevant_informations is None:
            return "No relevant information contained into the database."

        more_relevants_informations = [f"CHUNK {ids}:\n" + relevant_chunk for ids, relevant_chunk in
                                       enumerate(list_of_relevant_informations)]

        list_of_informations = "\n\n".join(more_relevants_informations)

        return list_of_informations

    # def retrieve_distances(self, strategy: str):
    #     # embed the user query
    #     query_embedded = self.embeds_query(query=strategy)
    #
    #     # search into the database
    #     results = self.client_collection.query(query_embeddings=query_embedded, n_results=50)
    #
    #     distances = results["distances"][0]
    #
    #     return distances  # [dis_1m dis_2, dis_3, ...]

    # def verify_strategies(self, strategies: str):
    #
    #     # retrieve possible informations from the different strategies
    #     # format the strategies like this: [**-**\n**-**]
    #     # to choose the best strategies calculate the distance, the nearest one is the best strategy
    #     best_distance = np.inf  # L2 distance goes from 0 to inf
    #     idx_strategy = 0
    #     list_of_strategies = strategies.split("**-**")
    #
    #     for index, strategy in enumerate(list_of_strategies):
    #         distances = self.retrieve_distances(strategy=strategy)
    #
    #         smallest_distance = np.min(distances)
    #
    #         if smallest_distance <= best_distance:
    #             best_distance = smallest_distance
    #             idx_strategy = index + 1
    #     # TODO after look to build ha rank of 1-2-3, so in the end all the strategy will be tried if needed
    #     if idx_strategy == 0:
    #         # The three strategies don't works, ask for others
    #         return None, False
    #
    #     return list_of_strategies[idx_strategy - 1], True

    # def similarity_score(self, distance: float) -> float:
    #     # transforming distance in range (0,1)
    #     score = 1 / (1 + distance)
    #
    #     return score

    def add_log(self, data) -> None:
        logger = logging.getLogger(__name__)
        # if ["prompt", "output", "feedback", "category"] not in data.keys():
        #    logger.error("missing data")

        # insert the feedback from the user inside the database
        vector = self.embeds_query(data['prompt'])
        # print(vector)
        self.db_log.insert(self.db_log_name, data, embeddings=vector)

    def rephrase_query(self, query: str) -> dict:
        """
        Rephrase user query for better achieving retrieval from a vector database and BM25
        """
        try:

            # define prompt
            prompt = """
            Given the user question, rephrase and expand it to help you do better answering. Maintain all information in the original question.
            This new reformulated query will be used to retrieve the most relevant information from our database.
            """
            # rephrase the query
            response = self.client_openai.responses.parse(
                model="gpt-4.1-mini",
                input=[{"role": "system", "content": prompt}, {"role": "user", "content": query}],
                text_format=RephraseOutput
            )
            # parsed_response = self.rephrase_parse_response(response.choices[0].message.content)
            # transform into a json object
            parsed_response = json.loads(response.output_text)

            return parsed_response
        except Exception as e:
            return {'intent': 'error', 'message': f"Failed to rephrase user query: {e}"}

    def split_query_in_keyword(self, query):
        """
        From the user query, extract keywords to use for searching into the BM25 database
        """
        try:

            prompt = """
            From the user query extract keywords to use for searching into the BM25 database
            Extract only the key technical terms, endpoint names, methods, and parameter names
            from the following query. 
            Return them as a short space-separated string, no explanations.
            """

            # retrieve the keyword of the query
            response = self.client_openai.responses.parse(
                model="gpt-4.1-mini",
                input=[{"role": "system", "content": prompt}, {"role": "user", "content": query}],
                text_format=ExtractKeywordOutput
            )
            # parsed_response = self.extract_keyword_response(response.choices[0].message.content)
            # transform into a json object
            parsed_response = json.loads(response.output_text)
            return parsed_response  # return a string -> transform it in a list using split

        except Exception as e:
            return {'intent': 'error', 'message': f"Failed to extract keyword: {e}"}

    # def rephrase_parse_response(self, response):
    #
    #     try:
    #         return RephraseOutput.model_validate_json(response)
    #     except Exception as e:
    #         return e
    #
    # def extract_keyword_response(self, response):
    #     try:
    #         return ExtractKeywordOutput.model_validate_json(response)
    #     except Exception as e:
    #         return e

    def rerank(self, query: str, chunk: str):
        """
        This function rerank the result obtained from hybrid search
        """
        try:
            print("Starting rerank")
            inputs = self.tokenizer(query, chunk, return_tensors="pt", truncation=True, max_length=512)
            self.model.eval()
            with torch.no_grad():
                scores = self.model(**inputs).logits

            return scores.item()

        except Exception as e:
            return f"Failed to rerank query: {e}"

    def rerank_and_add_to_list(self, merged_list: list, partial_result_list: list, keyword_field: str, query: str):
        """
        This function iterate through the partial list and rerank the result giving a new score
        """
        try:
            for hit in partial_result_list:
                score = self.rerank(query, hit[keyword_field])
                # add the score to the json object
                hit["score"] = score
                merged_list.append(hit)

            return merged_list

        except Exception as e:
            return f"Failed to rerank query: {e}"
