"""
RAG (Retrieval-Augmented Generation) system for LLM extraction.

Provides semantic search and context enhancement for extraction prompts.
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class EmbeddingGenerator:
    """
    Generate embeddings for clinical trial text.

    Uses medical-domain models for better semantic understanding.
    """

    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        device: str = "cpu"
    ):
        """
        Initialize embedding generator.

        Args:
            model_name: HuggingFace model name (medical-domain preferred)
            device: Device to use ('cpu' or 'cuda')
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.device = device
        self.model = None
        self.embedding_dim = 768  # Default for BERT-base models

    def load_model(self) -> None:
        """Load the embedding model (lazy loading)."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector as numpy array
        """
        self.load_model()
        return self.model.encode(text, convert_to_numpy=True)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            Embedding matrix (N x embedding_dim)
        """
        self.load_model()
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_document_chunks(
        self,
        chunks: List[Dict[str, str]],
        text_field: str = "text"
    ) -> np.ndarray:
        """
        Generate embeddings for document chunks.

        Args:
            chunks: List of chunk dictionaries
            text_field: Field name containing text to embed

        Returns:
            Embedding matrix
        """
        texts = [chunk.get(text_field, "") for chunk in chunks]
        return self.embed_texts(texts)


class VectorStore:
    """
    Vector store for semantic search using FAISS.

    Stores embeddings and enables fast similarity search.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        index_type: str = "flat"
    ):
        """
        Initialize vector store.

        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "faiss is required. Install with: pip install faiss-cpu"
            )

        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.documents: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None

    def create_index(self) -> None:
        """Create FAISS index."""
        if self.index_type == "flat":
            # Exact search (L2 distance)
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "ivf":
            # Inverted file index (faster for large datasets)
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
        elif self.index_type == "hnsw":
            # Hierarchical navigable small world (very fast)
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

    def add_documents(
        self,
        embeddings: np.ndarray,
        documents: List[Dict]
    ) -> None:
        """
        Add documents to the index.

        Args:
            embeddings: Embedding matrix (N x embedding_dim)
            documents: List of document metadata
        """
        if self.index is None:
            self.create_index()

        # Convert to float32 (required by FAISS)
        embeddings = embeddings.astype('float32')

        # For IVF, need to train first
        if self.index_type == "ivf" and not self.index.is_trained:
            self.index.train(embeddings)

        # Add to index
        self.index.add(embeddings)

        # Store documents
        self.documents.extend(documents)

        # Store embeddings
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            Tuple of (documents, distances)
        """
        if self.index is None or self.index.ntotal == 0:
            return [], np.array([])

        # Ensure correct shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype('float32')

        # Search
        distances, indices = self.index.search(query_embedding, k)

        # Get documents
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.documents):
                results.append(self.documents[idx])

        return results, distances[0]

    def save(self, path: str) -> None:
        """Save index and documents to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save index
        if self.index is not None:
            faiss.write_index(self.index, str(path / "index.faiss"))

        # Save documents
        with open(path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        # Save embeddings
        if self.embeddings is not None:
            np.save(path / "embeddings.npy", self.embeddings)

    def load(self, path: str) -> None:
        """Load index and documents from disk."""
        path = Path(path)

        # Load index
        if (path / "index.faiss").exists():
            self.index = faiss.read_index(str(path / "index.faiss"))

        # Load documents
        if (path / "documents.pkl").exists():
            with open(path / "documents.pkl", "rb") as f:
                self.documents = pickle.load(f)

        # Load embeddings
        if (path / "embeddings.npy").exists():
            self.embeddings = np.load(path / "embeddings.npy")


class Retriever:
    """
    Retrieve relevant examples for Few-shot learning.

    Combines semantic search with keyword matching.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator
    ):
        """
        Initialize retriever.

        Args:
            vector_store: Vector store for semantic search
            embedding_generator: Embedding generator
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator

    def retrieve(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Retrieve relevant examples.

        Args:
            query: Query text
            k: Number of examples to retrieve
            score_threshold: Optional maximum distance threshold

        Returns:
            List of retrieved examples with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.embed_text(query)

        # Search vector store
        results, distances = self.vector_store.search(query_embedding, k=k)

        # Filter by threshold if specified
        if score_threshold is not None:
            filtered = [
                (doc, dist) for doc, dist in zip(results, distances)
                if dist <= score_threshold
            ]
            results = [doc for doc, _ in filtered]
            distances = [dist for _, dist in filtered]

        # Add scores to results
        scored_results = []
        for doc, dist in zip(results, distances):
            doc_copy = doc.copy()
            doc_copy['relevance_score'] = float(dist)
            scored_results.append(doc_copy)

        return scored_results

    def retrieve_by_outcome(
        self,
        query: str,
        outcome_type: str,
        k: int = 5
    ) -> List[Dict]:
        """
        Retrieve examples filtered by outcome type.

        Args:
            query: Query text
            outcome_type: Type of outcome (binary, continuous, etc.)
            k: Number of examples to retrieve

        Returns:
            List of retrieved examples
        """
        # Retrieve with larger k
        all_results = self.retrieve(query, k=k * 2)

        # Filter by outcome type
        filtered = [
            r for r in all_results
            if r.get('outcome_type') == outcome_type
        ]

        return filtered[:k]


class PromptEnhancer:
    """
    Enhance prompts with retrieved examples.

    Adds Few-shot examples to extraction prompts.
    """

    def __init__(self, retriever: Retriever):
        """
        Initialize prompt enhancer.

        Args:
            retriever: Retriever for finding examples
        """
        self.retriever = retriever

    def enhance_prompt(
        self,
        base_prompt: str,
        study_context: Dict,
        n_examples: int = 3
    ) -> str:
        """
        Enhance prompt with retrieved examples.

        Args:
            base_prompt: Base prompt template
            study_context: Context about current study
            n_examples: Number of examples to add

        Returns:
            Enhanced prompt with examples
        """
        # Build query from study context
        query = self._build_query(study_context)

        # Retrieve relevant examples
        examples = self.retriever.retrieve(query, k=n_examples)

        if not examples:
            return base_prompt

        # Format examples
        examples_text = self._format_examples(examples)

        # Insert examples into prompt
        enhanced_prompt = self._insert_examples(base_prompt, examples_text)

        return enhanced_prompt

    def _build_query(self, study_context: Dict) -> str:
        """Build search query from study context."""
        parts = []

        if study_context.get('outcome'):
            parts.append(study_context['outcome'])

        if study_context.get('intervention'):
            parts.append(study_context['intervention'])

        if study_context.get('comparator'):
            parts.append(study_context['comparator'])

        return ' '.join(parts)

    def _format_examples(self, examples: List[Dict]) -> str:
        """Format retrieved examples as text."""
        formatted = []

        for i, example in enumerate(examples, 1):
            formatted.append(f"Example {i}:")
            formatted.append(f"Study: {example.get('study_title', 'Unknown')}")
            formatted.append(f"Outcome: {example.get('outcome', 'Unknown')}")
            formatted.append(f"Input: {example.get('input_text', '')[:200]}...")
            formatted.append(f"Output: {example.get('output', '')}")

            if example.get('relevance_score'):
                formatted.append(f"(Relevance: {1 - example['relevance_score']:.2f})")

            formatted.append("")

        return '\n'.join(formatted)

    def _insert_examples(self, prompt: str, examples_text: str) -> str:
        """Insert examples into prompt at appropriate location."""
        # Look for examples placeholder
        if '{examples}' in prompt:
            return prompt.replace('{examples}', examples_text)

        # If no placeholder, insert after instructions
        lines = prompt.split('\n')
        insert_idx = 0

        for i, line in enumerate(lines):
            if line.strip().lower().startswith('instructions:'):
                insert_idx = i + 1
                break
            if line.strip().lower().startswith('task:'):
                insert_idx = i + 1
                break

        if insert_idx > 0:
            lines.insert(insert_idx, '')
            lines.insert(insert_idx + 1, 'Here are some similar examples:')
            lines.insert(insert_idx + 2, examples_text)
            lines.insert(insert_idx + 3, '')

        return '\n'.join(lines)


def create_rag_system(
    data_path: str,
    model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    cache_dir: Optional[str] = None
) -> Tuple[EmbeddingGenerator, VectorStore, Retriever]:
    """
    Create a complete RAG system.

    Args:
        data_path: Path to indexed data
        model_name: Name of embedding model
        cache_dir: Directory for caching models

    Returns:
        Tuple of (embedding_generator, vector_store, retriever)
    """
    # Initialize components
    embedding_generator = EmbeddingGenerator(model_name=model_name)
    vector_store = VectorStore(embedding_dim=embedding_generator.embedding_dim)

    # Try to load existing index
    if cache_dir and Path(cache_dir).exists():
        try:
            vector_store.load(cache_dir)
        except Exception:
            pass

    retriever = Retriever(vector_store, embedding_generator)

    return embedding_generator, vector_store, retriever
