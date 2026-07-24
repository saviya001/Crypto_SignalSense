from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import time

class MessageType(str, Enum):
    REQUEST = "REQUEST"
    RESPONSE = "RESPONSE"
    ERROR = "ERROR"

class AgentRole(str, Enum):
    ROUTER = "router"
    NEWS_AGENT = "news_agent"
    TECHNICAL_AGENT = "technical_agent"
    SIGNAL_AGENT = "signal_agent"

class AgentMessage(BaseModel):
    """
    Structured Message Protocol for Agent-to-Agent Communication.
    Meets Section 4 (b) mandatory requirements for structured message exchange.
    """
    message_id: str = Field(default_factory=lambda: f"msg_{int(time.time()*1000)}")
    sender: AgentRole
    receiver: AgentRole
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: float = Field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes AgentMessage instance to a JSON-compatible dictionary format."""
        return {
            "message_id": self.message_id,
            "sender": self.sender.value,
            "receiver": self.receiver.value,
            "message_type": self.message_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp
        }
