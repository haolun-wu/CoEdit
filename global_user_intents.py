from typing import Dict, List, Set
from dataclasses import dataclass
from enum import Enum

class AtomicIntent(Enum):
    # Content Style
    CONCISE = "concise and to the point"
    DETAILED = "detailed and comprehensive"
    FORMAL = "formal and professional"
    CASUAL = "casual and conversational"
    
    # Structure
    STRUCTURED = "well-structured with clear sections"
    NARRATIVE = "narrative and flowing"
    BULLET_POINTS = "using bullet points for clarity"
    PARAGRAPH = "paragraph-based organization"
    
    # Focus
    TECHNICAL = "technical and precise"
    PRACTICAL = "practical and actionable"
    ANALYTICAL = "analytical and insightful"
    CREATIVE = "creative and innovative"
    
    # Purpose
    INFORMATIVE = "informative and educational"
    PERSUASIVE = "persuasive and convincing"
    COLLABORATIVE = "collaborative and inclusive"
    DECISIVE = "decisive and directive"

# Define the four perspectives and their corresponding intents
INTENT_PERSPECTIVES = {
    "Content Style": [AtomicIntent.CONCISE, AtomicIntent.DETAILED, AtomicIntent.FORMAL, AtomicIntent.CASUAL],
    "Structure": [AtomicIntent.STRUCTURED, AtomicIntent.NARRATIVE, AtomicIntent.BULLET_POINTS, AtomicIntent.PARAGRAPH],
    "Focus": [AtomicIntent.TECHNICAL, AtomicIntent.PRACTICAL, AtomicIntent.ANALYTICAL, AtomicIntent.CREATIVE],
    "Purpose": [AtomicIntent.INFORMATIVE, AtomicIntent.PERSUASIVE, AtomicIntent.COLLABORATIVE, AtomicIntent.DECISIVE]
}

@dataclass
class UserIntent:
    user_id: str
    intents: Set[AtomicIntent]

# Define user intents explicitly for each user
USER_INTENTS = {
    "user1": UserIntent("user1", {
        AtomicIntent.CONCISE,
        AtomicIntent.STRUCTURED,
        AtomicIntent.TECHNICAL,
        AtomicIntent.INFORMATIVE
    }),
    "user2": UserIntent("user2", {
        AtomicIntent.DETAILED,
        AtomicIntent.NARRATIVE,
        AtomicIntent.PRACTICAL,
        AtomicIntent.PERSUASIVE
    }),
    "user3": UserIntent("user3", {
        AtomicIntent.FORMAL,
        AtomicIntent.BULLET_POINTS,
        AtomicIntent.ANALYTICAL,
        AtomicIntent.COLLABORATIVE
    }),
    "user4": UserIntent("user4", {
        AtomicIntent.CASUAL,
        AtomicIntent.PARAGRAPH,
        AtomicIntent.CREATIVE,
        AtomicIntent.DECISIVE
    }),
    "user5": UserIntent("user5", {
        AtomicIntent.CONCISE,
        AtomicIntent.STRUCTURED,
        AtomicIntent.PRACTICAL,
        AtomicIntent.INFORMATIVE
    })
}

# Global guidelines for tasks
GLOBAL_GUIDELINES = {
    "summarization": {
        "cnn_dailymail": "style targeted to young children, storytelling, short sentences, playful language, interactive, positive",
        "slf5k": "second person narrative, brief, show emotions, invoke personal reflection, immersive",
        "wikipedia": "bullet points, parallel structure, brief",
        "CShorten/ML-ArXiv-Papers": "tweet style, simple English, inquisitive, skillful foreshadowing, with emojis",
        "imdb": "question answering style"
        },
    "email_writing": {
        "ccby": "structured, straight to the points, respectful, professional greeting and closing",
        "slf5k": "informal, conversational, no closing",
        "ampere": "casual tone, positive, clear, call to action",
        "paper_tweet": "engaging, personalized, professional tone, thankful closing"
        }
    }