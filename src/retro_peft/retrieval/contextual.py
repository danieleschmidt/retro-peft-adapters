"""
Contextual retrieval for conversation-aware and personalized retrieval.

Implements retrieval that considers conversation history, user context,
and temporal factors for improved relevance.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .retrievers import BaseRetriever


class ContextualRetriever:
    """
    Contextual retrieval wrapper that enhances base retrievers with:
    - Conversation history awareness
    - User personalization
    - Temporal decay
    - Query expansion
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        context_window: int = 5,
        context_weight: float = 0.4,
        temporal_decay: float = 0.9,
        query_expansion: bool = True,
        personalization_dim: int = 128,
    ):
        """
        Initialize contextual retriever.

        Args:
            base_retriever: Underlying retrieval backend
            context_window: Number of previous turns to consider
            context_weight: Weight for context in query fusion
            temporal_decay: Decay factor for older context
            query_expansion: Whether to expand queries with context
            personalization_dim: Dimension for user personalization vectors
        """
        self.base_retriever = base_retriever
        self.context_window = context_window
        self.context_weight = context_weight
        self.temporal_decay = temporal_decay
        self.query_expansion = query_expansion
        self.personalization_dim = personalization_dim

        # Conversation history storage
        self.conversation_history = []
        self.user_profiles = {}

        # Context encoder (use same as base retriever if available)
        self.encoder = base_retriever.encoder

        # Query expansion templates
        self.expansion_templates = [
            "Given the context of {context}, {query}",
            "Considering our previous discussion about {context}, {query}",
            "In relation to {context}, {query}",
            "Building on {context}, {query}",
        ]

    def add_to_conversation(
        self,
        query: str,
        response: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a conversation turn to history.

        Args:
            query: User query
            response: System response
            user_id: Optional user identifier
            metadata: Additional metadata for the turn
        """
        turn = {
            "query": query,
            "response": response,
            "user_id": user_id,
            "metadata": metadata or {},
            "timestamp": len(self.conversation_history),  # Simple timestamp
        }

        self.conversation_history.append(turn)

        # Keep only recent history
        if len(self.conversation_history) > self.context_window * 2:
            self.conversation_history = self.conversation_history[-self.context_window * 2 :]

        # Update user profile if user_id provided
        if user_id:
            self._update_user_profile(user_id, query, response)

    def _update_user_profile(self, user_id: str, query: str, response: str):
        """Update user personalization profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "interests": [],
                "query_patterns": [],
                "interaction_count": 0,
            }

        profile = self.user_profiles[user_id]
        profile["interaction_count"] += 1

        # Extract interests (simple keyword-based approach)
        # In practice, you might use more sophisticated methods
        keywords = self._extract_keywords(query + " " + response)
        profile["interests"].extend(keywords)

        # Keep only recent interests
        if len(profile["interests"]) > 50:
            profile["interests"] = profile["interests"][-50:]

        # Track query patterns
        profile["query_patterns"].append(query)
        if len(profile["query_patterns"]) > 20:
            profile["query_patterns"] = profile["query_patterns"][-20:]

    def _extract_keywords(self, text: str) -> List[str]:
        """Simple keyword extraction"""
        # Remove common stop words and extract meaningful terms
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
        }

        words = text.lower().split()
        keywords = [
            word.strip('.,!?;:"()[]{}')
            for word in words
            if len(word) > 3 and word.lower() not in stop_words
        ]

        # Return unique keywords
        return list(set(keywords))

    def retrieve_with_context(
        self,
        query: str,
        conversation_history: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        personalization_vector: Optional[np.ndarray] = None,
        k: int = 5,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Retrieve with conversation context and personalization.

        Args:
            query: Current query
            conversation_history: Optional conversation history
            user_id: Optional user identifier for personalization
            personalization_vector: Optional pre-computed user vector
            k: Number of documents to retrieve
            **kwargs: Additional arguments for base retriever

        Returns:
            Tuple of (context_embeddings, metadata_list)
        """
        # Use provided history or stored history
        if conversation_history is None:
            conversation_history = [
                turn["query"] for turn in self.conversation_history[-self.context_window :]
            ]

        # Build contextual query
        contextual_query = self._build_contextual_query(query, conversation_history, user_id)

        # Encode contextual query
        if self.encoder is None:
            raise ValueError("Encoder required for contextual retrieval")

        query_embedding = self.encoder.encode([contextual_query], convert_to_numpy=True)[0]

        # Apply personalization if available
        if personalization_vector is not None:
            query_embedding = self._apply_personalization(query_embedding, personalization_vector)
        elif user_id and user_id in self.user_profiles:
            user_vector = self._get_user_vector(user_id)
            query_embedding = self._apply_personalization(query_embedding, user_vector)

        # Retrieve using base retriever
        query_tensor = torch.from_numpy(query_embedding.reshape(1, -1))

        return self.base_retriever.retrieve(query_embeddings=query_tensor, k=k, **kwargs)

    def _build_contextual_query(
        self, query: str, conversation_history: List[str], user_id: Optional[str] = None
    ) -> str:
        """Build query enhanced with conversation context"""
        if not conversation_history or not self.query_expansion:
            return query

        # Extract key context from recent history
        context_text = self._extract_context(conversation_history)

        if not context_text:
            return query

        # Choose expansion template
        import random

        template = random.choice(self.expansion_templates)

        # Build contextual query
        contextual_query = template.format(context=context_text, query=query)

        # Add user-specific context if available
        if user_id and user_id in self.user_profiles:
            user_interests = self.user_profiles[user_id]["interests"][:5]  # Top interests
            if user_interests:
                interest_context = ", ".join(user_interests)
                contextual_query += f" (User interests: {interest_context})"

        return contextual_query

    def _extract_context(self, conversation_history: List[str]) -> str:
        """Extract key context from conversation history"""
        if not conversation_history:
            return ""

        # Simple approach: use last few queries
        recent_queries = conversation_history[-min(3, len(conversation_history)) :]

        # Extract key terms from recent queries
        all_keywords = []
        for query in recent_queries:
            keywords = self._extract_keywords(query)
            all_keywords.extend(keywords)

        # Get most frequent keywords
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1

        # Sort by frequency and take top terms
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        context_terms = [keyword for keyword, count in top_keywords]
        return " ".join(context_terms)

    def _get_user_vector(self, user_id: str) -> np.ndarray:
        """Get user personalization vector"""
        if user_id not in self.user_profiles:
            return np.zeros(self.personalization_dim)

        profile = self.user_profiles[user_id]

        # Simple approach: encode user interests
        if profile["interests"]:
            interest_text = " ".join(profile["interests"][:10])  # Top interests
            if self.encoder:
                user_vector = self.encoder.encode([interest_text], convert_to_numpy=True)[0]
                # Pad or truncate to personalization_dim
                if len(user_vector) > self.personalization_dim:
                    return user_vector[: self.personalization_dim]
                elif len(user_vector) < self.personalization_dim:
                    padded = np.zeros(self.personalization_dim)
                    padded[: len(user_vector)] = user_vector
                    return padded
                return user_vector

        return np.zeros(self.personalization_dim)

    def _apply_personalization(
        self, query_embedding: np.ndarray, personalization_vector: np.ndarray
    ) -> np.ndarray:
        """Apply personalization to query embedding"""
        # Simple approach: weighted combination
        personalization_weight = 0.2

        # Ensure compatible dimensions
        if len(personalization_vector) != len(query_embedding):
            # Resize personalization vector
            if len(personalization_vector) > len(query_embedding):
                personalization_vector = personalization_vector[: len(query_embedding)]
            else:
                padded = np.zeros(len(query_embedding))
                padded[: len(personalization_vector)] = personalization_vector
                personalization_vector = padded

        # Combine embeddings
        personalized_embedding = (
            1 - personalization_weight
        ) * query_embedding + personalization_weight * personalization_vector

        return personalized_embedding

    def retrieve_multi_turn(
        self,
        conversation: List[str],
        user_id: Optional[str] = None,
        k: int = 5,
        turn_weights: Optional[List[float]] = None,
        **kwargs,
    ) -> List[Tuple[torch.Tensor, List[Dict[str, Any]]]]:
        """
        Retrieve for multiple conversation turns with temporal weighting.

        Args:
            conversation: List of queries/turns
            user_id: Optional user identifier
            k: Number of documents per turn
            turn_weights: Optional weights for each turn
            **kwargs: Additional retriever arguments

        Returns:
            List of retrieval results for each turn
        """
        if turn_weights is None:
            # Apply temporal decay
            turn_weights = [
                self.temporal_decay ** (len(conversation) - i - 1) for i in range(len(conversation))
            ]

        results = []

        for turn_idx, query in enumerate(conversation):
            # Get context up to current turn
            context_history = conversation[:turn_idx]

            # Retrieve for this turn
            turn_results = self.retrieve_with_context(
                query=query, conversation_history=context_history, user_id=user_id, k=k, **kwargs
            )

            # Apply turn weight to scores
            context_embeddings, metadata_list = turn_results
            weight = turn_weights[turn_idx]

            # Weight the metadata scores
            weighted_metadata = []
            for meta in metadata_list:
                weighted_meta = meta.copy()
                if "score" in weighted_meta:
                    weighted_meta["score"] *= weight
                weighted_meta["turn_weight"] = weight
                weighted_metadata.append(weighted_meta)

            results.append((context_embeddings, weighted_metadata))

        return results

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation state"""
        return {
            "history_length": len(self.conversation_history),
            "context_window": self.context_window,
            "num_users": len(self.user_profiles),
            "recent_topics": self._get_recent_topics(),
            "user_stats": {
                user_id: {
                    "interactions": profile["interaction_count"],
                    "interests_count": len(profile["interests"]),
                }
                for user_id, profile in self.user_profiles.items()
            },
        }

    def _get_recent_topics(self) -> List[str]:
        """Extract recent conversation topics"""
        if not self.conversation_history:
            return []

        recent_queries = [turn["query"] for turn in self.conversation_history[-5:]]

        all_keywords = []
        for query in recent_queries:
            keywords = self._extract_keywords(query)
            all_keywords.extend(keywords)

        # Get most frequent keywords as topics
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1

        top_topics = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return [topic for topic, count in top_topics]

    def clear_conversation(self, user_id: Optional[str] = None):
        """Clear conversation history"""
        if user_id is None:
            self.conversation_history.clear()
        else:
            # Remove only turns from specific user
            self.conversation_history = [
                turn for turn in self.conversation_history if turn.get("user_id") != user_id
            ]

    def save_user_profiles(self, file_path: str):
        """Save user profiles to file"""
        import json

        with open(file_path, "w") as f:
            json.dump(self.user_profiles, f, indent=2)

    def load_user_profiles(self, file_path: str):
        """Load user profiles from file"""
        import json

        try:
            with open(file_path, "r") as f:
                self.user_profiles = json.load(f)
        except FileNotFoundError:
            print(f"Profile file not found: {file_path}")
        except json.JSONDecodeError:
            print(f"Invalid JSON in profile file: {file_path}")
