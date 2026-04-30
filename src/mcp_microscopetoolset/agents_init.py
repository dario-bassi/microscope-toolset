import os
import anthropic
from pymmcore_plus import CMMCorePlus
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from src.agentsNormal.specialized_agent import DatabaseAgent
from src.databases.elasticsearch_db import ElasticSearchDB
from src.local.execute import Execute
from src.mcp_microscopetoolset.utils import get_user_information, logger_database_exists
from src.microscope.microscope_status import MicroscopeStatus
from src.postqrl.connection import DBConnection
from src.postqrl.log_db import LoggerDB
import time
import logging
import sys

#  logger
logger = logging.getLogger("Initialize Agent")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    fh = logging.FileHandler("microscope_toolset.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)


def initialize_agents(mmc: CMMCorePlus, cfg_file: str | None = None):
    # Initialize the microscope session object
    logger.info("Initializing Microscope Session")
    logger.info("Microscope Session Initialized")

    # Get the information for the user
    logger.info("Getting User Information...")
    system_user_information = get_user_information()
    logger.info("System User Information: {}".format(system_user_information))

    # start executor and tracking of the microscope status
    logger.info("Initializing Executor...")
    executor = Execute(mmc=mmc, filename=cfg_file)
    logger.info("Executor Initialized")
    logger.info("Initializing Microscope Status...")
    microscope_status = MicroscopeStatus(executor=executor)
    logger.info("Microscope Status Initialized")
    # initialize Logger database and his connection (optional — requires DB_* env-vars)
    logger.info("Initializing Logger database...")
    if os.getenv("DB_HOST") and os.getenv("DB_NAME") and os.getenv("DB_USER") and os.getenv("DB_PASSWORD") and os.getenv("DB_PORT"):
        try:
            db_connection = DBConnection()
            db_log = LoggerDB(db_connection)
            logger.info("Logger database initialized")
            if not logger_database_exists(db_log, system_user_information['log_collection']):
                db_log.create_collection(system_user_information['log_collection'])
                logger.info(f"A new collection named {system_user_information['log_collection']} has been created.")
        except Exception as e:
            logger.warning(f"PostgreSQL initialization failed: {e}. Continuing without logger database.")
            db_log = None
    else:
        logger.info("DB_HOST not set — skipping PostgreSQL logger database.")
        db_log = None

    # Try to connect to Elasticsearch (optional — skip entirely when not configured)
    es_client = None
    database_agent = None
    es_path = system_user_information.get('elastic_search_path_home')
    if not es_path:
        logger.info("ELASTICSEARCH not set in .env — skipping database tools.")
    else:
        try:
            es_url = system_user_information.get('elasticsearch_url')
            es_client = ElasticSearchDB(url=es_url)
            logger.info("Initialed ElasticSearch Python Client")
            max_retries = 5
            retry_delay = 1  # seconds

            for attempt in range(max_retries):
                logger.info(f"Trying connection to Elasticsearch (attempt {attempt + 1}/{max_retries})...")
                if es_client.is_connected():
                    logger.info("Connected to Elasticsearch!")
                    break
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 8)
                es_client = ElasticSearchDB(url=es_url)
            else:
                logger.warning("Could not connect to Elasticsearch. Database tools will be unavailable.")
                es_client = None

            if es_client is not None:
                # get relevant information for the db
                pdf_publication = system_user_information.get('pdf_collection_name', '')
                micromanager_collection = system_user_information.get('micromanager_devices_collection', '')
                api_collection = system_user_information.get('collection_name', '')
                logger.info(f"Database Name: {pdf_publication}")
                logger.info(f"Micromanager Collection: {micromanager_collection}")
                logger.info(f"API Collection: {api_collection}")

                # Load the cross-encoder for re-ranking
                reranker_name = "cross-encoder/ms-marco-MiniLM-L6-v2"
                tokenizer = AutoTokenizer.from_pretrained(reranker_name)
                reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_name)
                logger.info(f"Cross-encoder model {reranker_name} loaded")

                # Load the sentence-transformers embedding model.
                # NOTE: dimension must match the ES KNN index — re-index if you change model.
                embed_model_name = system_user_information.get('embed_model', 'BAAI/bge-small-en-v1.5')
                embed_model = SentenceTransformer(embed_model_name)
                logger.info(f"Embedding model {embed_model_name!r} loaded")

                # Anthropic client — only created when the API key is present.
                # Without it, query reformulation is skipped but ES search still works.
                anthropic_model = system_user_information.get('anthropic_model', 'claude-haiku-4-5-20251001')
                if os.getenv("ANTHROPIC_API_KEY"):
                    client = anthropic.Anthropic()
                    logger.info(f"Anthropic client loaded (model: {anthropic_model!r})")
                else:
                    client = None
                    logger.warning("ANTHROPIC_API_KEY not set — query reformulation will be unavailable")

                # initialize different Agents
                database_agent = DatabaseAgent(
                    client=client,
                    es_client=es_client,
                    pdf_collection=pdf_publication,
                    micromanager_collection=micromanager_collection,
                    api_collection=api_collection,
                    db_log=db_log,
                    db_log_collection_name=system_user_information.get('log_collection', ''),
                    tokenizer=tokenizer,
                    model=reranker_model,
                    embed_model=embed_model,
                    llm_model=anthropic_model,
                )
                logger.info("Initialed Database Agent")
        except Exception as e:
            logger.warning(f"Elasticsearch/Database agent initialization failed: {e}. Continuing without database tools.")
            es_client = None
            database_agent = None

    return {
        "executor": executor,
        "microscope_status": microscope_status,
        "database_agent": database_agent,
        "db_log": db_log,
        "es_client": es_client,
    }




