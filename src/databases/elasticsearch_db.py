from elasticsearch import Elasticsearch


class ElasticSearchDB:

    def __init__(self) -> None:

        self.es = Elasticsearch("http://localhost:4500")
        #self.index_name = index_name

    def is_connected(self) -> bool:
        """
        This function returns True if the connection is established.
        """
        return self.es.ping()

    def close(self) -> None:
        """
        Disconnect python Client from server
        """
        self.es.close()

        print("The client was disconnected")

    def shutdown(self) -> None:
        self.es.shutdown()

    def get_info(self):
        """
        This function gets the server information from Elasticsearch
        """
        api_answer = self.es.info()

        return api_answer


    def create_default_index(self, index_name: str, **kwargs):
        """
        This function creates a default index in the Elasticsearch database
        """

        properties = {}

        for key, value in kwargs.items():
            properties.update({key: value})

        # add embedding field
        properties["embedding"] = {
            "type": "dense_vector",
            "dims": 512,
            "similarity": "l2_norm",
            "index": True,
            "index_options": {
                "type": "hnsw",
                "m": 32,
                "ef_construction": 100
            }
        }

        index_settings = {
            "settings": {
                "similarity": {"default": {"type": "BM25"}}
            },
            "mappings": {
                "properties": properties
            }
        }

        if not self.es.indices.exists(index=index_name):
            api_answer = self.es.indices.create(index=index_name, body=index_settings)
            print("Index created!")
            return api_answer
        else:
            print("Index already exists!")

            return "Index already exists!"

    def index_documents(self, index_name: str, documents: list[dict[str, ...]]):
        """
        This function indexes documents
        """
        action = []
        for doc in documents:
            source = {}
            for key, value in doc.items():
                source.update({key: value})

            action.append({"index": {"_index": index_name}})
            action.append(source)

        api_answer = self.es.bulk(operations=action)
        # refresh the index for searching
        self.es.indices.refresh(index=index_name)

        return api_answer

    def search(self, index_name: str, queries: str, k_result: int = 50):
        """
        This function queries the Elasticsearch database and retrieve the k-best results
        """
        # refresh indices before searching
        self.es.indices.refresh(index=index_name)
        # TODO think is multi_match is more appropriate
        search_body = {
            "query": {
                "match": {
                    "query": queries,
                }
            },
            "size": k_result
        }
        search_result = self.es.search(index=index_name, body=search_body)

        return search_result

    def delete_index(self, index_name: str):
        """
        This function allows to delete an index/collection from the Elasticsearch database
        """
        api_answer = self.es.indices.delete(index=index_name)
        print("Index deleted!")
        return api_answer

    def hybrid_search(self, index_name: str, query: str, query_vector: list[float], keyword_to_search_for_bm25: str, k_result: int = 150, semantic_weight: float = 0.8,
                      bm25_weight: float = 0.2):
        """
        This function retrieve the best results obtained by semantic search and keywords search (BM25)
        """
        # refresh indices before searching
        self.es.indices.refresh(index=index_name)

        search_body = {
            "query": {
                "match": {
                    keyword_to_search_for_bm25: {
                        "query": query,
                        "boost": semantic_weight
                    }
                }
            },
            "knn": {
                "field": "embedding",
                "query_vector": query_vector,
                "k": k_result,
                "num_candidates": 2 * k_result,
                "boost": bm25_weight
            },
            "size": k_result
        }

        api_answer = self.es.search(index=index_name, body=search_body)

        return api_answer
