import json
from openai import OpenAI
import  logging
import sys

#  logger
logger = logging.getLogger("BaseAgent")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    fh = logging.FileHandler("microscope_toolset.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

class BaseAgent:

    def __init__(self, client_openai: OpenAI):
        self.client_openai = client_openai


    def call_agent(self, model: str, input_user: list, error_string: str, output_format):
        """
        This function declare how the agent is called
        """
        try:
            response = self.client_openai.responses.parse(
                model=model,
                input=input_user,
                text_format=output_format
            )

            # parse json object
            parsed_response = json.loads(response.output_text)
            logger.info(parsed_response)
            return parsed_response
        except Exception as e:
            logger.error({"intent": "error", "message": f"{error_string}: {e}"})
            return {"intent": "error", "message": f"{error_string}: {e}"}
