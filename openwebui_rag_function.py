"""
title: Personal Health Agent RAG
description: RAG function for Personal Health Agent research papers
author: open-webui
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 1.0.0
license: MIT
required_open_webui_version: 0.6.0
"""

import os
import sys
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional, Callable, Awaitable

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vector_store import VectorStore


class Filter:
    pass


class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Awaitable[None]] = None):
        self.event_emitter = event_emitter

    async def emit(self, description="Unknown State", status="in_progress", done=False):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
            )


class Tools:
    class Valves(BaseModel):
        TOP_K: int = Field(default=5, description="Number of documents to retrieve")
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.citation = True

        # Initialize vector store
        try:
            self.vector_store = VectorStore()
            self.vector_store.create_collection(reset=False)
            print(f"[RAG Function] Vector store initialized with {self.vector_store.collection.count()} documents")
        except Exception as e:
            print(f"[RAG Function] Error initializing vector store: {e}")
            self.vector_store = None

    async def search_papers(
        self,
        query: str,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> str:
        """
        Search Personal Health Agent research papers and return relevant contexts.

        :param query: The search query
        :return: Relevant paper excerpts
        """
        emitter = EventEmitter(__event_emitter__)

        if not self.vector_store:
            await emitter.emit(
                description="Vector store not initialized",
                status="error",
                done=True
            )
            return "Error: Vector store not available"

        try:
            await emitter.emit(
                description=f"Searching papers for: {query}",
                status="in_progress"
            )

            # Search vector store
            results = self.vector_store.search(query, k=self.valves.TOP_K)

            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]

            if not documents:
                await emitter.emit(
                    description="No relevant papers found",
                    status="complete",
                    done=True
                )
                return "No relevant information found in the research papers."

            # Format results
            context = "## Relevant Research Paper Excerpts:\n\n"

            for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), 1):
                relevance = round(1 - dist, 3)
                source = meta.get('source', 'Unknown')

                context += f"### [{i}] {source} (Relevance: {relevance})\n"
                context += f"{doc}\n\n"

            await emitter.emit(
                description=f"Found {len(documents)} relevant excerpts",
                status="complete",
                done=True
            )

            return context

        except Exception as e:
            await emitter.emit(
                description=f"Error searching papers: {str(e)}",
                status="error",
                done=True
            )
            return f"Error: {str(e)}"
