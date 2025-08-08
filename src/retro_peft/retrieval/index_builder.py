"""
Vector index builder for creating retrieval indices from document corpora.
"""

import hashlib
import json
import os
from typing import Any, Callable, Dict, List, Optional, Union

import h5py
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


class VectorIndexBuilder:
    """
    Build vector indices from document corpora for retrieval augmentation.

    Supports multiple embedding models and efficient index construction
    with caching and metadata extraction.
    """

    def __init__(
        self,
        encoder: Union[str, SentenceTransformer] = "sentence-transformers/all-mpnet-base-v2",
        chunk_size: int = 512,
        overlap: int = 50,
        device: str = "auto",
    ):
        """
        Initialize vector index builder.

        Args:
            encoder: Embedding model name or instance
            chunk_size: Maximum tokens per chunk
            overlap: Overlap tokens between chunks
            device: Device for computation ("auto", "cpu", "cuda")
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Setup device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Initialize encoder
        if isinstance(encoder, str):
            try:
                # Try SentenceTransformer first
                self.encoder = SentenceTransformer(encoder, device=self.device)
                self.encoder_type = "sentence_transformer"
            except:
                # Fall back to HuggingFace transformers
                self.tokenizer = AutoTokenizer.from_pretrained(encoder)
                self.encoder = AutoModel.from_pretrained(encoder).to(self.device)
                self.encoder_type = "transformers"
        else:
            self.encoder = encoder
            self.encoder_type = "sentence_transformer"

        self.embedding_dim = self._get_embedding_dim()

    def _get_embedding_dim(self) -> int:
        """Get embedding dimension from encoder"""
        if self.encoder_type == "sentence_transformer":
            return self.encoder.get_sentence_embedding_dimension()
        else:
            # Test with dummy input
            dummy_input = self.tokenizer("test", return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.encoder(**dummy_input)
            return output.last_hidden_state.size(-1)

    def chunk_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.

        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of chunks with text and metadata
        """
        if self.encoder_type == "sentence_transformer":
            # Use simple word-based chunking
            words = text.split()
            chunks = []

            for i in range(0, len(words), self.chunk_size - self.overlap):
                chunk_words = words[i : i + self.chunk_size]
                chunk_text = " ".join(chunk_words)

                chunk_data = {
                    "text": chunk_text,
                    "start_idx": i,
                    "end_idx": min(i + self.chunk_size, len(words)),
                    "metadata": metadata or {},
                }
                chunks.append(chunk_data)

                if i + self.chunk_size >= len(words):
                    break

        else:
            # Use tokenizer-based chunking
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            chunks = []

            for i in range(0, len(tokens), self.chunk_size - self.overlap):
                chunk_tokens = tokens[i : i + self.chunk_size]
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)

                chunk_data = {
                    "text": chunk_text,
                    "start_idx": i,
                    "end_idx": min(i + self.chunk_size, len(tokens)),
                    "metadata": metadata or {},
                }
                chunks.append(chunk_data)

                if i + self.chunk_size >= len(tokens):
                    break

        return chunks

    def encode_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 32) -> np.ndarray:
        """
        Encode text chunks into embeddings.

        Args:
            chunks: List of chunk dictionaries
            batch_size: Batch size for encoding

        Returns:
            Embedding matrix of shape [num_chunks, embedding_dim]
        """
        texts = [chunk["text"] for chunk in chunks]
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            if self.encoder_type == "sentence_transformer":
                batch_embeddings = self.encoder.encode(
                    batch_texts, convert_to_numpy=True, show_progress_bar=False
                )
            else:
                # HuggingFace transformers encoding
                inputs = self.tokenizer(
                    batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.encoder(**inputs)
                    # Mean pooling
                    attention_mask = inputs["attention_mask"]
                    embeddings_tensor = outputs.last_hidden_state
                    input_mask_expanded = (
                        attention_mask.unsqueeze(-1).expand(embeddings_tensor.size()).float()
                    )
                    sum_embeddings = torch.sum(embeddings_tensor * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()

            embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def build_index(
        self,
        documents: List[Union[str, Dict[str, Any]]],
        metadata_extractor: Optional[Callable] = None,
        embedding_cache: Optional[str] = None,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Build vector index from documents.

        Args:
            documents: List of documents (strings or dicts with 'text' field)
            metadata_extractor: Optional function to extract metadata
            embedding_cache: Optional path to cache embeddings
            batch_size: Batch size for encoding

        Returns:
            Dictionary with embeddings, chunks, and metadata
        """
        print(f"Building index from {len(documents)} documents...")

        # Process documents into chunks
        all_chunks = []
        for doc_idx, doc in enumerate(documents):
            if isinstance(doc, str):
                text = doc
                doc_metadata = {"doc_id": doc_idx}
            else:
                text = doc.get("text", str(doc))
                doc_metadata = doc.get("metadata", {})
                doc_metadata["doc_id"] = doc_idx

            # Extract additional metadata if provided
            if metadata_extractor:
                extracted_metadata = metadata_extractor(doc)
                doc_metadata.update(extracted_metadata)

            # Chunk document
            chunks = self.chunk_text(text, doc_metadata)
            all_chunks.extend(chunks)

        print(f"Created {len(all_chunks)} chunks")

        # Check for cached embeddings
        embeddings = None
        if embedding_cache and os.path.exists(embedding_cache):
            cache_hash = self._compute_cache_hash(all_chunks)
            try:
                with h5py.File(embedding_cache, "r") as f:
                    if f.attrs.get("hash") == cache_hash:
                        embeddings = f["embeddings"][:]
                        print("Loaded embeddings from cache")
            except:
                print("Cache file corrupted, recomputing embeddings")

        # Compute embeddings if not cached
        if embeddings is None:
            print("Computing embeddings...")
            embeddings = self.encode_chunks(all_chunks, batch_size)

            # Save to cache if specified
            if embedding_cache:
                os.makedirs(os.path.dirname(embedding_cache), exist_ok=True)
                cache_hash = self._compute_cache_hash(all_chunks)
                with h5py.File(embedding_cache, "w") as f:
                    f.create_dataset("embeddings", data=embeddings)
                    f.attrs["hash"] = cache_hash
                    f.attrs["embedding_dim"] = self.embedding_dim
                    f.attrs["num_chunks"] = len(all_chunks)
                print(f"Saved embeddings to cache: {embedding_cache}")

        return {
            "embeddings": embeddings,
            "chunks": all_chunks,
            "embedding_dim": self.embedding_dim,
            "num_chunks": len(all_chunks),
        }

    def _compute_cache_hash(self, chunks: List[Dict[str, Any]]) -> str:
        """Compute hash for cache validation"""
        content = json.dumps([chunk["text"] for chunk in chunks], sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def export_faiss(self, index_data: Dict[str, Any], output_path: str) -> str:
        """
        Export index to FAISS format.

        Args:
            index_data: Index data from build_index
            output_path: Output path for FAISS index

        Returns:
            Path to saved FAISS index
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS not installed. Run: pip install faiss-cpu")

        embeddings = index_data["embeddings"]

        # Create FAISS index
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # Use IVF index for large collections, flat for small
        if len(embeddings) > 10000:
            nlist = min(int(np.sqrt(len(embeddings))), 1000)
            quantizer = faiss.IndexFlatIP(embeddings.shape[1])
            index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], nlist)
            index.train(embeddings)
        else:
            index = faiss.IndexFlatIP(embeddings.shape[1])

        # Add embeddings
        index.add(embeddings)

        # Save index
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        faiss.write_index(index, output_path)

        # Save metadata
        metadata_path = output_path.replace(".faiss", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "chunks": index_data["chunks"],
                    "embedding_dim": index_data["embedding_dim"],
                    "num_chunks": index_data["num_chunks"],
                },
                f,
                indent=2,
            )

        print(f"FAISS index saved to: {output_path}")
        return output_path

    def export_qdrant(self, index_data: Dict[str, Any], collection_url: str) -> str:
        """
        Export index to Qdrant.

        Args:
            index_data: Index data from build_index
            collection_url: Qdrant collection URL

        Returns:
            Collection name
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, PointStruct, VectorParams
        except ImportError:
            raise ImportError("Qdrant client not installed. Run: pip install qdrant-client")

        # Parse URL
        if "://" in collection_url:
            host = collection_url.split("://")[1].split("/")[0]
            collection_name = collection_url.split("/")[-1]
        else:
            host = "localhost:6333"
            collection_name = collection_url

        client = QdrantClient(host=host)

        # Create collection
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=index_data["embedding_dim"], distance=Distance.COSINE),
        )

        # Prepare points
        points = []
        for i, (embedding, chunk) in enumerate(zip(index_data["embeddings"], index_data["chunks"])):
            point = PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={"text": chunk["text"], "metadata": chunk["metadata"]},
            )
            points.append(point)

        # Upload in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            client.upsert(collection_name=collection_name, points=batch)

        print(f"Qdrant collection created: {collection_name}")
        return collection_name

    def export_weaviate(self, index_data: Dict[str, Any], schema_url: str) -> str:
        """
        Export index to Weaviate.

        Args:
            index_data: Index data from build_index
            schema_url: Weaviate schema URL

        Returns:
            Class name
        """
        try:
            import weaviate
        except ImportError:
            raise ImportError("Weaviate client not installed. Run: pip install weaviate-client")

        # Parse URL
        if "://" in schema_url:
            host = schema_url.split("://")[1].split("/")[0]
            class_name = schema_url.split("/")[-1]
        else:
            host = "localhost:8080"
            class_name = schema_url

        client = weaviate.Client(f"http://{host}")

        # Create schema
        schema = {
            "class": class_name,
            "vectorizer": "none",
            "properties": [
                {"name": "text", "dataType": ["text"]},
                {"name": "metadata", "dataType": ["object"]},
            ],
        }

        # Delete existing class if it exists
        try:
            client.schema.delete_class(class_name)
        except:
            pass

        client.schema.create_class(schema)

        # Upload data in batches
        batch_size = 100
        for i in range(0, len(index_data["chunks"]), batch_size):
            batch_chunks = index_data["chunks"][i : i + batch_size]
            batch_embeddings = index_data["embeddings"][i : i + batch_size]

            with client.batch as batch:
                for chunk, embedding in zip(batch_chunks, batch_embeddings):
                    batch.add_data_object(
                        data_object={"text": chunk["text"], "metadata": chunk["metadata"]},
                        class_name=class_name,
                        vector=embedding.tolist(),
                    )

        print(f"Weaviate class created: {class_name}")
        return class_name
