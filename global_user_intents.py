from typing import Dict, List, Set
from dataclasses import dataclass
from enum import Enum

class AtomicIntent(Enum):
    # --- Content Style ---
    CONCISE = "eliminate redundancy and make the text short and direct"
    EXPANSIVE = "expand ideas with supporting details and examples"
    FORMAL = "use academic or professional tone, avoiding casual language"
    FRIENDLY = "use conversational tone, including contractions and approachable phrasing"

    # --- Structure ---
    ORGANIZED = "clearly segment content into logical sections or headings"
    STORY_LIKE = "present content as a flowing narrative or sequence of events"
    BULLETED = "convert lists or ideas into bullet points for scannability"
    PARAGRAPHIC = "keep ideas grouped into well-developed paragraphs"

    # --- Focus ---
    TECHNICAL = "emphasize precision, jargon, and specificity suitable for experts"
    HANDS_ON = "prioritize examples, instructions, and direct applicability"
    CRITICAL = "analyze, question, or evaluate the subject rigorously"
    INVENTIVE = "include imaginative ideas or novel perspectives"

    # --- Purpose ---
    TEACHING = "make content clear and educational, suitable for learning"
    INFLUENCING = "persuade the reader toward a viewpoint or decision"
    BRAINSTORMING = "encourage open-ended, inclusive, or divergent thinking"
    DECISION_MAKING = "present arguments and conclude with actionable guidance"


@dataclass
class UserIntent:
    user_id: str
    intents: Set[AtomicIntent]

# Define user intents explicitly for each user
USER_INTENTS = {
    "user1": UserIntent("user1", {
        AtomicIntent.CONCISE,
        AtomicIntent.ORGANIZED,
        AtomicIntent.TECHNICAL,
        AtomicIntent.TEACHING
    }),
    "user2": UserIntent("user2", {
        AtomicIntent.EXPANSIVE,
        AtomicIntent.STORY_LIKE,
        AtomicIntent.HANDS_ON,
        AtomicIntent.INFLUENCING
    }),
    "user3": UserIntent("user3", {
        AtomicIntent.FORMAL,
        AtomicIntent.BULLETED,
        AtomicIntent.CRITICAL,
        AtomicIntent.DECISION_MAKING
    }),
    "user4": UserIntent("user4", {
        AtomicIntent.FRIENDLY,
        AtomicIntent.PARAGRAPHIC,
        AtomicIntent.INVENTIVE,
        AtomicIntent.BRAINSTORMING
    }),
    "user5": UserIntent("user5", {
        AtomicIntent.CONCISE,
        AtomicIntent.ORGANIZED,
        AtomicIntent.HANDS_ON,
        AtomicIntent.TEACHING
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