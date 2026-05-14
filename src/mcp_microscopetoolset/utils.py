import os

from dotenv import load_dotenv
from mcp.types import Tool

# from agentsNormal.classify_user_intent import ClassifyAgent
# from agentsNormal.structuredOutput import ClassificationAgentOutput
from postqrl import LoggerDB


def user_message(message):
    return {"role": "user", "content": message}


def agent_message(message: str):
    return {"role": "assistant", "content": message}


def tool_message(tool_call_id: str, tool_name: str, content: str):
    return {"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": content}


def _add_to_conversation(context: dict, role: str, message: str):
    """Helper to add messages to the conversation history within the context."""
    context["conversation"].append({"role": role, "content": message})


def mcp_to_openai(tool: Tool):
    """
    Transform a mcp tool into an openai function schema
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
            "strict": False,
        },
    }


# add initialization components
def get_user_information() -> dict:
    """
    This function retrieves the user information from the system. Later we will add the possibility for the user to add it
    """
    user_information = {}
    # load user from the environment
    load_dotenv()

    user_information["collection_name"] = os.getenv("DBNAME")
    user_information["log_collection"] = os.getenv("LOGNAME")
    user_information["cfg_file"] = os.getenv("CFGPATH")
    user_information["pdf_collection_name"] = os.getenv("PDFDB")
    user_information["micromanager_devices_collection"] = os.getenv("DEVDB")
    user_information["elastic_search_path_home"] = os.getenv("ELASTICSEARCH")
    user_information["elasticsearch_url"] = os.getenv("ELASTICSEARCH_URL", "http://localhost:4500")
    user_information["fastmcp_server_path"] = os.getenv("FASTMCP_SERVER")
    user_information["benchmark_agent_enable"] = os.getenv("BENCHMARK_KNOWLEDGE_ENABLED")
    user_information["anthropic_model"] = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    user_information["embed_model"] = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    user_information["proxy_core_host"] = os.getenv("PROXY_CORE_HOST", "127.0.0.1")
    user_information["proxy_core_port"] = os.getenv("PROXY_CORE_PORT", "5601")

    return user_information


def logger_database_exists(logger: LoggerDB, name: str) -> bool:
    """
    This function check if a logger database exists
    """

    # list the connection in the database
    list_of_collection = logger.list_collection()

    if name in list_of_collection:
        # The collection is already present
        return True
    else:
        # The collection doesn't exist
        return False
