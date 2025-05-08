from typing import Dict, List, Set
from dataclasses import dataclass
from enum import Enum

class AtomicIntent(Enum):
    # Content Style
    FORMAL = "formal"  # Professional, academic tone
    CASUAL = "casual"  # Conversational, friendly tone
    TECHNICAL = "technical"  # Detailed, specialized language
    SIMPLE = "simple"  # Easy to understand, clear language
    
    # Structure
    STRUCTURED = "structured"  # Clear organization, bullet points
    NARRATIVE = "narrative"  # Story-like flow
    ANALYTICAL = "analytical"  # Logical analysis, evidence-based
    
    # Focus
    DETAILED = "detailed"  # Comprehensive coverage
    CONCISE = "concise"  # Brief, to the point
    ENGAGING = "engaging"  # Interactive, attention-grabbing
    
    # Purpose
    EDUCATIONAL = "educational"  # Teaching, learning focus
    PERSUASIVE = "persuasive"  # Convincing, influential
    INFORMATIVE = "informative"  # Factual, objective information

# Define task-specific intent combinations
TASK_INTENTS = {
    "summarization": {
        "cnn_dailymail": {
            "C1": {AtomicIntent.SIMPLE, AtomicIntent.STRUCTURED, AtomicIntent.CONCISE},
            "C2": {AtomicIntent.FORMAL, AtomicIntent.ANALYTICAL, AtomicIntent.DETAILED},
            "C3": {AtomicIntent.CASUAL, AtomicIntent.NARRATIVE, AtomicIntent.ENGAGING},
            "C4": {AtomicIntent.TECHNICAL, AtomicIntent.STRUCTURED, AtomicIntent.INFORMATIVE},
            "C5": {AtomicIntent.SIMPLE, AtomicIntent.NARRATIVE, AtomicIntent.EDUCATIONAL}
        },
        "slf5k": {
            "C1": {AtomicIntent.CASUAL, AtomicIntent.NARRATIVE, AtomicIntent.ENGAGING},
            "C2": {AtomicIntent.FORMAL, AtomicIntent.ANALYTICAL, AtomicIntent.DETAILED},
            "C3": {AtomicIntent.SIMPLE, AtomicIntent.STRUCTURED, AtomicIntent.INFORMATIVE},
            "C4": {AtomicIntent.TECHNICAL, AtomicIntent.ANALYTICAL, AtomicIntent.EDUCATIONAL},
            "C5": {AtomicIntent.CASUAL, AtomicIntent.NARRATIVE, AtomicIntent.PERSUASIVE}
        }
    },
    "email_writing": {
        "slf5k": {
            "C1": {AtomicIntent.FORMAL, AtomicIntent.STRUCTURED, AtomicIntent.PERSUASIVE},
            "C2": {AtomicIntent.CASUAL, AtomicIntent.NARRATIVE, AtomicIntent.ENGAGING},
            "C3": {AtomicIntent.TECHNICAL, AtomicIntent.ANALYTICAL, AtomicIntent.INFORMATIVE},
            "C4": {AtomicIntent.SIMPLE, AtomicIntent.STRUCTURED, AtomicIntent.EDUCATIONAL},
            "C5": {AtomicIntent.FORMAL, AtomicIntent.ANALYTICAL, AtomicIntent.DETAILED}
        }
    }
}

@dataclass
class UserPreference:
    user_id: str
    task_preferences: Dict[str, Dict[str, Set[AtomicIntent]]]  # task -> dataset -> intent combination

# Define user preferences using combinations
USER_PREFERENCES = {
    "user1": UserPreference(
        user_id="user1",
        task_preferences={
            "summarization": {
                "cnn_dailymail": TASK_INTENTS["summarization"]["cnn_dailymail"]["C1"],
                "slf5k": TASK_INTENTS["summarization"]["slf5k"]["C1"]
            },
            "email_writing": {
                "slf5k": TASK_INTENTS["email_writing"]["slf5k"]["C1"]
            }
        }
    ),
    "user2": UserPreference(
        user_id="user2",
        task_preferences={
            "summarization": {
                "cnn_dailymail": TASK_INTENTS["summarization"]["cnn_dailymail"]["C2"],
                "slf5k": TASK_INTENTS["summarization"]["slf5k"]["C2"]
            },
            "email_writing": {
                "slf5k": TASK_INTENTS["email_writing"]["slf5k"]["C2"]
            }
        }
    ),
    "user3": UserPreference(
        user_id="user3",
        task_preferences={
            "summarization": {
                "cnn_dailymail": TASK_INTENTS["summarization"]["cnn_dailymail"]["C3"],
                "slf5k": TASK_INTENTS["summarization"]["slf5k"]["C3"]
            },
            "email_writing": {
                "slf5k": TASK_INTENTS["email_writing"]["slf5k"]["C3"]
            }
        }
    ),
    "user4": UserPreference(
        user_id="user4",
        task_preferences={
            "summarization": {
                "cnn_dailymail": TASK_INTENTS["summarization"]["cnn_dailymail"]["C4"],
                "slf5k": TASK_INTENTS["summarization"]["slf5k"]["C4"]
            },
            "email_writing": {
                "slf5k": TASK_INTENTS["email_writing"]["slf5k"]["C4"]
            }
        }
    ),
    "user5": UserPreference(
        user_id="user5",
        task_preferences={
            "summarization": {
                "cnn_dailymail": TASK_INTENTS["summarization"]["cnn_dailymail"]["C5"],
                "slf5k": TASK_INTENTS["summarization"]["slf5k"]["C5"]
            },
            "email_writing": {
                "slf5k": TASK_INTENTS["email_writing"]["slf5k"]["C5"]
            }
        }
    )
}

# Global guidelines for each task
GLOBAL_GUIDELINES = {
    "summarization": {
        "cnn_dailymail": "You are an AI assistant helping users understand news articles. Focus on clarity, accuracy, and engaging storytelling while maintaining journalistic standards.",
        "slf5k": "You are an AI assistant helping users understand personal stories. Focus on clarity and key insights while maintaining the original message."
    },
    "email_writing": {
        "slf5k": "You are an AI assistant helping users write professional emails. Focus on clarity, professionalism, and effective communication."
    }
} 