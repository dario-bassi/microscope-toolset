import logging
from src.postqrl.connection import DBConnection
from pgvector.psycopg2 import register_vector
from typing import List

logger = logging.getLogger(__name__)
from psycopg2.extras import Json


class LoggerDB:

    def __init__(self, connection: DBConnection):
        self.connection = connection
        self.initialize()

    def initialize(self):
        connection = self.connection.get_connect()

        try:
            with connection.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                connection.commit()
                print("Vector extension was activated")
                register_vector(cur)
                print("pgvector type adapter REGISTERED.")
        except Exception as e:
            logger.error(e)
        finally:
            self.connection.put_connect(connection)

    def list_collection(self) -> list[str]:
        connection = self.connection.get_connect()
        try:
            with connection.cursor() as cur:
                cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                """)
                return [row[0] for row in cur.fetchall()]
        except Exception as e:
            logger.error(e)
            return []
        finally:
            self.connection.put_connect(connection)

    def create_collection(self, collection_name: str):
        connection = self.connection.get_connect()
        # verify that the name doesn't exits already
        if collection_name in self.list_collection():
            logger.error("The database contains already a collection called %s", collection_name)
        try:
            with connection.cursor() as cur:
                cur.execute(f"""
                CREATE TABLE {collection_name}(
                id SERIAL PRIMARY KEY,
                prompt TEXT NOT NULL,
                output TEXT NOT NULL,
                feedback BOOLEAN,
                category TEXT,
                embedding vector(512),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW()
                );
                """)
                # add microscope configuration, microscope state, reformulated prompt and solution in text way, summarize/optmization the strategy from the coversation.
                connection.commit()
                logger.info(f"The collection {collection_name} was created successfully")
        except Exception as e:
            logger.error(e)

        finally:
            self.connection.put_connect(connection)

    def get_collection(self, collection_name: str) -> list[dict]:
        connection = self.connection.get_connect()

        if collection_name not in self.list_collection():
            logger.error(f"The collection {collection_name} doesn't exist or is not present into the database.")

        try:
            with connection.cursor() as cur:
                cur.execute(f"""
                SELECT * FROM {collection_name}
                """)
                return [
                    {"prompt": row[1], "output": row[2], "feedback": row[3], "category": row[4], "embedding": row[5]}
                    for row in cur.fetchall()]
        except Exception as e:
            logger.error(e)
            return []
        finally:
            self.connection.put_connect(connection)

    def get_columns_name(self, collection_name: str) -> list:
        connection = self.connection.get_connect()

        if collection_name not in self.list_collection():
            logger.error(f"The collection {collection_name} doesn't exist or is not present into the database.")

        try:
            with connection.cursor() as cur:
                cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s
                """, (collection_name,))
                return cur.fetchall()
        except Exception as e:
            logger.error(e)
            return []
        finally:
            self.connection.put_connect(connection)

    def update_collection(self, collection_name: str, **update):
        connection = self.connection.get_connect()

        if collection_name not in self.list_collection():
            logger.error(f"The collection {collection_name} doesn't exist or is not present into the database.")

        if len(update) == 0:
            logger.error("No updated values available.")

        if update.keys() not in self.get_columns_name(collection_name):
            logger.error("One or more column name are not present in the collection.")

        try:
            with connection.cursor() as cur:
                for i in update:
                    value_to_update = update[i]
                    cur.execute(f"""
                    UPDATE %s
                    SET %s = {value_to_update}
                    """, (collection_name, i))
                    cur.commit()
                    logger.info(f"The value of {i} was changed in {value_to_update}")
        except Exception as e:
            logger.info(e)
        finally:
            self.connection.put_connect(connection)

    def insert(self, collection_name: str, data_dict, embeddings=List[float]):

        connection = self.connection.get_connect()

        if collection_name not in self.list_collection():
            logger.error(f"The collection {collection_name} is not in the database")

        try:

            with connection.cursor() as cur:
                register_vector(cur)
                print("cursor run in insertion")
                cur.execute(f"""
                INSERT INTO {collection_name}
                (prompt, output, feedback, category, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                data_dict["prompt"], data_dict["output"], data_dict["feedback"], data_dict["category"],
                embeddings, Json(data_dict)))

                connection.commit()
                print("Vector inserted successfully using custom adapter!")

        except Exception as e:
            logger.error(e)
        finally:
            self.connection.put_connect(connection)

    def delete(self, collection_name: str):

        connection = self.connection.get_connect()

        if collection_name not in self.list_collection():
            logger.error(f"The collection {collection_name} is not into the database")

        try:
            with connection.cursor() as cur:
                cur.execute(f"""
                DROP TABLE IF EXISTS {collection_name}
                """)

                logger.info(f"The collection {collection_name} was permanently deleted.")

        except Exception as e:
            logger.error(e)

        finally:
            self.connection.put_connect(connection)


    def query(self):

        pass

    def query_by_category(self, collection_name: str, category: str, k: int = 10) -> list[dict]:
        connection = self.connection.get_connect()

        if collection_name not in self.list_collection():
            logger.error(f"The collection {collection_name} is not present into the database")

        try:
            with connection.cursor() as cur:
                cur.execute(f"""
                SELECT * FROM {collection_name} WHERE category = %s LIMIT %s
                """, (category, k))
                return [
                    {"prompt": row[1], "output": row[2], "feedback": row[3], "category": row[4], "embedding": row[5]}
                    for row in cur.fetchall()]
        except Exception as e:
            logger.error(e)
            return []
        finally:
            self.connection.put_connect(connection)

    def query_by_vector(self, collection_name: str, vector: list[float], k: int = 10) -> list[dict]:
        connection = self.connection.get_connect()

        if collection_name not in self.list_collection():
            logger.error(f"The collection {collection_name} is not present into the database")
        try:
            with connection.cursor() as cur:
                register_vector(cur)
                cur.execute(f"""
                SELECT prompt, output, feedback, category, (embedding <-> %s::vector) AS distance, embedding
                FROM {collection_name}
                ORDER BY distance
                LIMIT %s
                """, (vector, k))
                return [
                    {"prompt": row[0], "output": row[1], "feedback": row[2], "category": row[3], "distance": row[4]}
                    for row in cur.fetchall()]
        except Exception as e:
            logger.error(e)
            return []
        finally:
            self.connection.put_connect(connection)

    def query_feedback(self, collection_name: str, feedback: bool, k: int = 10) -> list[dict]:
        connection = self.connection.get_connect()

        if collection_name not in self.list_collection():
            logger.error(f"The collection {collection_name} is not into the database.")
        try:
            with connection.cursor() as cur:
                cur.execute(f"""
                SELECT * FROM {collection_name} WHERE feedback = %s LIMIT %s
                """, (feedback, k))
                return [
                    {"prompt": row[1], "output": row[2], "feedback": row[3], "category": row[4], "embedding": row[5]}
                    for row in cur.fetchall()]
        except Exception as e:
            logger.error(e)
            return []
        finally:
            self.connection.put_connect(connection)


    def close(self):

        try:
            self.connection.disconnect()
        except Exception as e:
            logger.error(e)

