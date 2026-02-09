from pydantic import BaseModel


class RephraseOutput(BaseModel):
    user_query: str
    reformulated_query: str

class ExtractKeywordOutput(BaseModel):
    keyword: list[str]

