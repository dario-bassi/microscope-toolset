import os
import logging
from typing import Optional
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class DBConnection:
    def __init__(self, db_host: Optional[str] = None, db_name: Optional[str] = None, db_user: Optional[str] = None,
                 db_password: Optional[str] = None, db_port: Optional[int] = None):

        load_dotenv()

        self.db_host = db_host or os.getenv("DB_HOST")
        self.db_name = db_name or os.getenv("DB_NAME")
        self.db_user = db_user or os.getenv("DB_USER")
        self.db_password = db_password or os.getenv("DB_PASSWORD")

        if db_port is not None:
            self.db_port = db_port
        else:
            raw_port = os.getenv("DB_PORT")
            if raw_port is None:
                raise ValueError("DB_PORT is not set in environment.")
            try:
                self.db_port = int(raw_port)
            except ValueError:
                raise ValueError(f"DB_PORT must be an integer, got: {raw_port!r}")


        #print(self.db_port,self.db_password,self.db_user,self.db_name, self.db_host)

        self.pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=30,
            host=self.db_host,
            user=self.db_user,
            database=self.db_name,
            password=self.db_password,
            port=self.db_port)

    def get_connect(self):
        return self.pool.getconn()

    def put_connect(self, connection):
        self.pool.putconn(connection)

    def disconnect(self):
        self.pool.closeall()