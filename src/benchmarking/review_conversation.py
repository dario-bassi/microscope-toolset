import json
from typing import Any, Literal


def review_experiment_conversation(path: str):
    """Review the conversation and make a dynamic summary of the conversation with a dashboard"""

    # check the path
    if path is None:
        raise FileExistsError("The path you included doesn't exists.")
    

    # start the dashboard with the file
    pass


def read_file(path: str) -> list[Any]:
    """Read the JSONL file with the full conversation."""
    conversation = []

    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line)
                # Extract specific fields
                conversation.append(data)
            except json.JSONDecodeError:
                continue  # Skip invalid lines
    # close file streaming
    file.close()

    return conversation


def generic_message(generic_message: dict[Any, Any]):
    pass


def user_message(user_dict: dict[Any, Any]):
    """Assign user dictonary messages."""

    type_var: Literal['user', 'assistant', 'file-history-snapshot', 'queue-operation'] = user_dict['type']
    uuid_var: str = user_dict['uuid']
    parentUuid_var: str = user_dict['parentUuid']
    sessionId_var: str = user_dict['sessionId']
    isSideChain_var: bool = user_dict['isSideChain_var']
    isMeta_var: bool = user_dict['isMeta']
    userType_var: str = user_dict['userType']
    cwd_var: str = user_dict['cwd']
    gitBranch_var: str = user_dict['gitBranch']
    timeStamp_var: str = user_dict['timeStamp']
    thinkingMetadata_var: dict = user_dict['thinkingMetadata']
    toolUsesMessages_var: list = user_dict['toolUsesMessages']
    todos_var = user_dict['todos']
    slug_var: str = user_dict['slug']
    requestId_var: str = user_dict['requestId']
    toolUseResult_var = user_dict['toolUseResult']
    agentId_var: str = user_dict['agentId']
    content_var: str | list[Any] = user_dict['message']['content'] if isinstance(user_dict['message']['content'], list) else user_dict['message']['content']
    toolUseId_toolContent: str | None = user_dict['message']['content'][0]['tool_use_id'] if isinstance(user_dict['message']['content'], list) else None
    content_toolContent: str | None = user_dict['message']['content'][0]['content'] if isinstance(user_dict['message']['content'], list) else None
    isError_toolContent: bool | None = user_dict['message']['content'][0]['is_error'] if isinstance(user_dict['message']['content'], list) else None



def assistant_message(assistant_dict: dict[Any, Any]):
    """Assign assistant dictonary messages."""

    type_var: Literal['user', 'assistant', 'file-history-snapshot', 'queue-operation'] = assistant_dict['type']
    uuid_var: str = assistant_dict['uuid']
    parentUuid_var: str = assistant_dict['parentUuid']
    sessionId_var: str = assistant_dict['sessionId']
    isSideChain_var: bool = assistant_dict['isSideChain_var']
    isMeta_var: bool = assistant_dict['isMeta']
    userType_var: str = assistant_dict['userType']
    cwd_var: str = assistant_dict['cwd']
    gitBranch_var: str = assistant_dict['gitBranch']
    timeStamp_var: str = assistant_dict['timeStamp']
    thinkingMetadata_var: dict = assistant_dict['thinkingMetadata']
    toolUsesMessages_var: list = assistant_dict['toolUsesMessages']
    todos_var = assistant_dict['todos']
    slug_var: str = assistant_dict['slug']
    requestId_var: str = assistant_dict['requestId']
    toolUseResult_var = assistant_dict['toolUseResult']
    agentId_var: str = assistant_dict['agentId']
    content_var: str | list[Any] = assistant_dict['message']['content'] if isinstance(assistant_dict['message']['content'], list) else assistant_dict['message']['content']
    id_toolUse: str | None = assistant_dict['message']['content'][0]['id'] if isinstance(assistant_dict['message']['content'], list) and assistant_dict['message']['content'][0]['type'] == 'tool_use' else None
    name_toolUse: str | None = assistant_dict['message']['content'][0]['name'] if isinstance(assistant_dict['message']['content'], list) and assistant_dict['message']['content'][0]['type'] == 'tool_use' else None
    input_toolUse: dict | None = assistant_dict['message']['content'][0]['input'] if isinstance(assistant_dict['message']['content'], list) and assistant_dict['message']['content'][0]['type'] == 'tool_use' else None
    thinking_thinking: str | None = assistant_dict['message']['content'][0]['thinking'] if isinstance(assistant_dict['message']['content'], list) and assistant_dict['message']['content'][0]['type'] == 'thinking' else None
    signature_thinking: str | None = assistant_dict['message']['content'][0]['signature'] if isinstance(assistant_dict['message']['content'], list) and assistant_dict['message']['content'][0]['type'] == 'thinking' else None
    text_text: str | None = assistant_dict['message']['content'][0]['text'] if isinstance(assistant_dict['message']['content'], list) and assistant_dict['message']['content'][0]['type'] == 'text' else None


def file_snap_message(file_snap_dict: dict[Any, Any]):
    """Assign file snapshot messages."""

    type_var: Literal['user', 'assistant', 'file-history-snapshot', 'queue-operation'] = file_snap_dict['type']
    messageId_var: str = file_snap_dict['messageId']
    snapshot_var: dict = file_snap_dict['snapshot']
    isSnapshotUpdated_var: bool = file_snap_dict['isSnapshotUpdate']



def queue_operation_message(queue_operation_dict: dict[Any, Any]):
    """Assign queue_operation_messages"""

    type_var: Literal['user', 'assistant', 'file-history-snapshot', 'queue-operation'] = queue_operation_dict['type']
    operation_var: Literal['enqueue', 'dequeue'] = queue_operation_dict['operation']
    timestamp_var: str = queue_operation_dict['timestamp']
    sessionId_var: str = queue_operation_dict['sessionId']
    content_var: str = queue_operation_dict['content']