
from enum import Enum
from pydantic import BaseModel

class Role(str, Enum):
    USER:str='user'
    SYSTEM:str='system'
    ASSITANT:str='assistant'

class Message(BaseModel):
    role:Role
    content:str