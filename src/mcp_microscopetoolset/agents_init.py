import os
from openai import OpenAI
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.experimental.unicore import UniMMCore
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.agentsNormal.logger_agent import LoggerAgent
from src.agentsNormal.classify_user_intent import ClassifyAgent
from src.agentsNormal.specialized_agent import DatabaseAgent, SoftwareEngeneeringAgent, StrategyAgent, NoCodingAgent
from src.databases.elasticsearch_db import ElasticSearchDB
from src.local.execute import Execute
from src.mcp_microscopetoolset.microscope_session import MicroscopeSession
from src.mcp_microscopetoolset.utils import logger_database_exists, get_user_information
from src.microscope.microscope_status import MicroscopeStatus
from src.postqrl.connection import DBConnection
from src.postqrl.log_db import LoggerDB
import time
import logging
import sys

#  logger
logger = logging.getLogger("Initialize Agent")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)
fh = logging.FileHandler("microscope_toolset.log", encoding="utf-8")
fh.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logger.addHandler(fh)


def initialize_agents(mmc: CMMCorePlus | UniMMCore, microscope_type: str = "real"):
    # Initialize the microscope session object
    logger.info("Initializing Microscope Session")
    microscope_session_object = MicroscopeSession()
    logger.info("Microscope Session Initialized")
    # create the data_dict that will contain the feedback loop information
    # data_dict = microscope_session_object.get_data_dict()

    # Get the information for the user
    logger.info("Getting User Information...")
    system_user_information = get_user_information()
    logger.info("System User Information: {}".format(system_user_information))
    # Determine configuration file based on executor
    if microscope_type == "virtual":
        cfg_file = None
        logger.info("Initializing UniCore...")
    else:
        cfg_file = system_user_information['cfg_file']
        logger.info(f"Initializing real microscope with config: {cfg_file}")
    # start executor and tracking of the microscope status
    logger.info("Initializing Executor...")
    executor = Execute(filename=cfg_file, mmc=mmc, microscope_type=microscope_type)
    logger.info("Executor Initialized")
    logger.info("Initializing Microscope Status...")
    microscope_status = MicroscopeStatus(executor=executor)
    logger.info("Microscope Status Initialized")
    # initialize Logger database and his connection
    logger.info("Initializing Logger database...")
    logger.info("Initializing connection...")
    #db_connection = DBConnection() # for the moment comment this part for testing
    #db_log = LoggerDB(db_connection)

    # check if the logger database already exist
    #logger.info("Checking if logger database exists...")
    #if not logger_database_exists(db_log, system_user_information['log_collection']):
        # it doesn't exist. We create a new one
    #    db_log.create_collection(system_user_information['log_collection'])
    #    logger.info(f"A new collection named {system_user_information['log_collection']} has been created.")

    es_client = ElasticSearchDB()
    logger.info("Initialed ElasticSearch Python Client")
    max_retries = 100
    retry_delay = 1  # seconds

    for attempt in range(max_retries):
        logger.info(f"Trying connection to Elasticsearch (attempt {attempt + 1}/{max_retries})...")
        if es_client.is_connected():
            logger.info("Connected to Elasticsearch!")
            break
        time.sleep(retry_delay)
        retry_delay *= 2
        es_client = ElasticSearchDB()
    else:
        logger.error("Could not connect to Elasticsearch after 100 attempts. Please make sure the server is running.")
        raise RuntimeError(
            "Could not connect to Elasticsearch after 100 attempts. Please make sure the server is running.")

    # get relevant information for the db
    # --NEW-- for the moment comment this part for testing
    #pdf_publication = system_user_information['pdf_collection_name']
    #micromanager_collection = system_user_information['micromanager_devices_collection']
    api_collection = system_user_information['collection_name']
    #logger.info(f"Database Name: {pdf_publication}")
    #logger.info(f"Micromanager Collection: {micromanager_collection}")
    logger.info(f"API Collection: {api_collection}")

    # Load the cross-encoder for re-ranking
    # Load model directly
    model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    logger.info(f"Cross-encoder Model {model_name} loaded")

    # initialize LLM API
    client_openai = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    logger.info("LLM API loaded")

    # initialize different Agents
    database_agent = DatabaseAgent(client_openai=client_openai, es_client=es_client, pdf_collection=None,#pdf_publication
                                   micromanager_collection=None, api_collection=api_collection, # micromanager_collection
                                   db_log=None, db_log_collection_name=system_user_information['log_collection'], # db_log
                                   tokenizer=tokenizer, model=model)
    logger.info("Initialed Database Agent")

    software_agent = SoftwareEngeneeringAgent(client_openai=client_openai)
    logger.info("Initialed Software Agent")

    strategy_agent = StrategyAgent(client_openai=client_openai)
    logger.info("Initialed Strategy Agent")

    no_coding_agent = NoCodingAgent(client_openai=client_openai)
    logger.info("Initialed NoCoding Agent")

    logger_agent = LoggerAgent(client_openai=client_openai)
    logger.info("Initialed Logger Agent")

    classify_agent = ClassifyAgent(client_openai=client_openai)
    logger.info("Initialed Classify Agent")

    return {
        "microscope_session_object": microscope_session_object,
        "executor": executor,
        "microscope_status": microscope_status,
        "database_agent": database_agent,
        "software_agent": software_agent,
        "strategy_agent": strategy_agent,
        "no_coding_agent": no_coding_agent,
        "logger_agent": logger_agent,
        "classify_agent": classify_agent,
        "db_log": None,#db_log
        "es_client": es_client,
        "client_openai": client_openai,
    }




