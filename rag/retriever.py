import time
import sys
import pypdf
import docx
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Add the project root to sys.path before trying to import custom packages
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import get_api_key, get_model_configuration, get_parameter_configuration
_qdrant_clients: Dict[str, Any] = {}


def _get_qdrant_client(qdrant_path: Path) -> QdrantClient:
    """
    Return a QdrantClient for the given storage path, reusing an existing
    instance when possible.

    Qdrant's local (on-disk) mode locks the database directory. Creating
    multiple clients for the same path causes lock errors. This helper
    keeps a module-level cache so the same path always returns the same
    client instance.

    Args:
        qdrant_path: Filesystem path where the Qdrant data is stored.

    Returns:
        A QdrantClient connected to that path.
    """
    key = str(qdrant_path.resolve())
    if key not in _qdrant_clients:
        _qdrant_clients[key] = QdrantClient(path=key)
    return _qdrant_clients[key]


class MedicalKnowladgeRetriever:
    def __init__(
                self,
                docs_dir = None,
                collection_name = None, 
                chunk_size = None,
                overlap = None,
                reset_cache = False 
                ):

        """
        Initialize all arguments here related to RAG pipeline
        """
        self.docs_dir = docs_dir or get_parameter_configuration().get("path", {}).get("docs_directory", {})
        self.collection_name = collection_name or get_parameter_configuration().get("rag", {}).get("collection_name", {})
        self.chunk_size = chunk_size or get_parameter_configuration().get("chunking", {}).get("chunk_size", {})
        self.overlap = overlap or get_parameter_configuration().get("chunking", {}).get("chunk_overlap", {})
        self.reset_cache = reset_cache or get_parameter_configuration().get("path", {}).get("cache_dir", {})

        parameter_config = get_parameter_configuration()
        rag_config = parameter_config.get("rag", {})
        retrieval_config = parameter_config.get("retrieval", {})

        self.embed_model = rag_config.get("embed_model", "text-embedding-3-small")
        self.max_k = retrieval_config.get("top_k", 4)
        self.score_threshold = rag_config.get("score_threshold", 0.18)
        self.openai_client = OpenAI(api_key=get_api_key("OPENAI"))

        self.store_path = Path(rag_config.get("cache_path", "./store"))
        if not self.store_path.is_absolute():
            self.store_path = ROOT_DIR / self.store_path

        qdrant_path = self.store_path / "qdrant"
        qdrant_path.mkdir(parents=True, exist_ok=True)
        self._client = _get_qdrant_client(qdrant_path)
        # self._is_indexed = False
        self._is_indexed = self._client.count(self.collection_name).count > 0
        self._ensure_collection()


    def _ensure_collection(self, reset_cache: bool = False):
        """
        Ensure the vector collection exists in the client, optionally resetting it.

        If `reset_cache` is True and the collection already exists, it is deleted first.
        Then, if the collection is missing (either because it never existed or was just
        deleted), a new collection is created with the configured embedding size and
        cosine distance metric.
        """
        if reset_cache and self._client.collection_exists(self.collection_name):
            self._client.delete_collection(self.collection_name)
        if not self._client.collection_exists(self.collection_name):
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                                            size=1536,
                                            distance=Distance.COSINE,
                                            ),
                                        )

    def _read_txt(self, file_path: Path):
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read().replace("\x00", "").strip()
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            return ""

    def _read_pdf(self, file_path: Path) -> str:
        if pypdf is None:
            print(f"PDF support not available (install pypdf). Skipping {file_path.name}")
            return ""
        try:
            reader = pypdf.PdfReader(str(file_path))
            parts = []
            for i, page in enumerate(reader.pages):
                t = page.extract_text()
                if t:
                    parts.append(f"\n--- Page {i + 1} ---\n{t}")
            return "".join(parts) if parts else ""
        except Exception as e:
            print(f"Error reading PDF {file_path.name}: {e}")
            return ""

    def _read_docx(self, file_path: Path) -> str:
        if docx is None:
            print(f"DOCX support not available (install python-docx). Skipping {file_path.name}")
            return ""
        try:
            doc = docx.Document(str(file_path))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            print(f"Error reading DOCX {file_path.name}: {e}")
            return ""

    def _extract_text_from_file(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            return self._read_txt(file_path)
        if suffix == ".pdf":
            return self._read_pdf(file_path)
        if suffix == ".docx":
            return self._read_docx(file_path)
        print(f"Unsupported format {suffix} for {file_path.name}")
        return ""

    def _chunk_text(self, text, source):
        """
        Split a long text into overlapping chunks while trying to respect sentence boundaries.

        Args:
            text: The full text to chunk.
            source: Identifier of the document this text came from (stored in every chunk).

        Returns:
            A list of dictionaries, each containing:
            - "text": the chunk content
            - "source": the original source identifier
            - "chunk_id": sequential integer ID of the chunk
        """
        chunks = []
        text = text.strip()
        if not text:
            return chunks

        start = 0
        chunk_id = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            # Try to break at a sentence boundary if we are not at the very end of the text
            if end < len(text):
                # Only look in the last ~150 characters of the current chunk
                search_start = max(0, len(chunk_text) - 150)
                for delimiter in [". ", ".\n", "? ", "?\n", "! ", "!\n"]:
                    last_pos = chunk_text.rfind(delimiter, search_start)
                    if last_pos > 0:
                        # Truncate the chunk so it ends cleanly after the sentence
                        chunk_text = chunk_text[: last_pos + len(delimiter)]
                        end = start + last_pos + len(delimiter)
                        break

            chunks.append({
                            "text": chunk_text.strip(),
                            "source": source,
                            "chunk_id": chunk_id,
                            })

            if end >= len(text):
                break
            start = end - self.overlap

            if start < 0:
                start = 0

            chunk_id += 1

        return chunks

    def _get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the given text using the configured OpenAI model.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        response = self.openai_client.embeddings.create(model=self.embed_model, input=text)
        return response.data[0].embedding

    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generates vector embeddings for a batch of input texts using OpenAI.

        Sanitizes inputs by handling non-string types, empty inputs, null bytes,
        and excessive length before making a batched API request.

        Args:
            texts (List[str]): A list of string inputs to be embedded.

        Returns:
            List[List[float]]: A list of floating-point vector embeddings, 
                matching the order and size of the input list.
        """
        if not texts:
            return []

        max_char_limit = 30000
        safe_texts: List[str] = []

        for text in texts:
            # Handle non string values
            if not isinstance(text, str):
                safe_texts.append(" ")
                continue
            # Remove null bytes (causes API/DB issues)
            cleaned_text = text.replace("\x00", "")

            # Handle empty strings (OpenAI rejects empty inputs)
            cleaned_text = cleaned_text.strip()
            if not cleaned_text:
                cleaned_text = " "

            # Truncate string to prevent token/payload limits 
            safe_texts.append(cleaned_text[:max_char_limit])

        # Batch call to OpenAI API
        response = self.openai_client.embeddings.create(
                                                        model=self.embed_model, 
                                                        input=safe_texts
                                                        )

        # Reconstruct matrix of float vectors preserving index order
        return [[float(val) for val in item.embedding] for item in response.data]

    def _collect_document_paths(self) -> List[Path]:
        """ Scans the designated directory for supported document formats. """
        paths = []
        supported_extensions = {".txt", ".pdf", ".docx"}

        for ext in supported_extensions:
            paths.extend(self.docs_directory.glob(f"*{ext}"))
        return sorted(paths)   

    def index_documents(self, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Parses, embeds, and stores documents from local storage into Qdrant.

        Extracts text from supported formats, breaks text into manageable chunks,
        generates vector embeddings via OpenAI in batches of 20, and upserts 
        PointStruct records containing vector indices and text payload to Qdrant.

        Args:
            force_reindex (bool): If True, resets existing vector collection and 
                re-indexes all files regardless of cache state. Defaults to False.

        Returns:
            Dict[str, Any]: Status summary containing operational counts, indexing status,
                and total processing elapsed execution time.
        """
        if self._is_indexed and not force_reindex:
            return {
                "status": "already_indexed",
                "document_count": self._client.count(self.collection_name).count,
            }
        if force_reindex:
            self._ensure_collection(reset_cache=True)
            self._is_indexed = False

        start_time = time.time()
        doc_paths = self._collect_document_paths()
        if not doc_paths:
            return {"status": "no_documents", "document_count": 0}

        print(f"Found {len(doc_paths)} document(s) to index (.txt, .pdf, .docx)...")
        all_chunks = []
        for path in doc_paths:
            print(f"Processing {path.name}...")
            text = self._extract_text_from_file(path)
            if text:
                all_chunks.extend(self._chunk_text(text, path.stem))

        all_chunks = [c for c in all_chunks if (c.get("text") or "").strip()]
        if not all_chunks:
            return {"status": "no_content", "document_count": 0}

        n_chunks = len(all_chunks)
        print(f"Created {n_chunks} chunks. Embedding and saving to Qdrant (store/)...")

        batch_size = 20
        for i in range(0, n_chunks, batch_size):
            batch = all_chunks[i : i + batch_size]
            texts = [
                str((c.get("text") or "").strip() or " ").replace("\x00", "")
                for c in batch
            ]
            embeddings = self._get_embeddings_batch(texts)
            points = [
                PointStruct(
                    id=i + j,
                    vector=embeddings[j],
                    payload={
                        "text": texts[j],
                        "source": str(c["source"]),
                        "chunk_id": int(c["chunk_id"]),
                    },
                )
                for j, c in enumerate(batch)
            ]
            self._client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            print(f"  {min(i + batch_size, n_chunks)}/{n_chunks} chunks")

        self._is_indexed = True
        elapsed = time.time() - start_time
        print(f"Done in {elapsed:.1f}s.")
        return {
                "status": "indexed",
                "document_count": len(doc_paths),
                "chunk_count": n_chunks,
                "elapsed_seconds": round(elapsed, 2),
                }
    
    def retrieve(
                    self,
                    query: str,
                    max_k: Optional[int] = None,
                    score_threshold: Optional[float] = None,
                    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant text chunks for a query from the vector store.

        Args:
            query: The search query text.
            max_k: Maximum number of results to return. Falls back to self.max_k if not provided.
            score_threshold: Minimum similarity score required. Falls back to self.score_threshold.

        Returns:
            A list of dictionaries, each containing:
            - "text": the retrieved chunk text
            - "source": the original document source
            - "chunk_id": the chunk identifier
            - "similarity": the cosine similarity score (rounded to 4 decimals)
        """
        if not self._is_indexed:
            return []

        k = max_k or self.max_k
        threshold = score_threshold or self.score_threshold

        query_embedding = self._get_embedding(query)

        response = self._client.query_points(
                                            collection_name=self.collection_name,
                                            query=query_embedding,
                                            limit=k
                                            )

        # Normalize different possible response formats from the Qdrant client
        if response is None:
            hits = []
        elif hasattr(response, "points"):
            hits = response.points
        elif isinstance(response, list):
            hits = response
        else:
            hits = list(response) if hasattr(response, "__iter__") else []

        # Qdrant COSINE distance returns similarity scores (higher = more similar)
        out = []
        for h in hits:
            sim = float(h.score) if h.score is not None else 0.0
            if sim >= threshold and h.payload:
                out.append({
                    "text": h.payload.get("text", ""),
                    "source": h.payload.get("source", "unknown"),
                    "chunk_id": h.payload.get("chunk_id"),
                    "similarity": round(sim, 4),
                })

        return out

    def get_context(
                    self,
                    query: str,
                    max_k: Optional[int] = None,
                    include_metadata: bool = True,
                    ) -> str:
        """
        Retrieve relevant chunks and format them into a single context string
        suitable for injecting into an LLM prompt.

        Args:
            query: The search query.
            max_k: Maximum number of chunks to include. Uses self.max_k if omitted.
            include_metadata: If True, prepends source information to each chunk.

        Returns:
            A formatted string containing the retrieved context, or a fallback
            message if nothing relevant was found.
        """
        chunks = self.retrieve(query, max_k=max_k)
        if not chunks:
            return "No relevant information found in internal knowledge base."

        parts = [
            f"[Source {idx}: {chunk['source']}]\n{chunk['text']}"
            if include_metadata
            else chunk["text"]
            for idx, chunk in enumerate(chunks, 1)
        ]

        return "\n\n---\n\n".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        """
        Return basic statistics about the current state of the knowledge base.

        Returns:
            A dictionary containing:
            - indexed: whether any data has been indexed
            - total_chunks: number of vectors currently stored
            - chunk_size / overlap: the chunking parameters in use
            - embed_model: the embedding model name
            - collection_name: the Qdrant collection being used
        """
        count = self._client.count(self.collection_name).count
        return {
            "indexed": self._is_indexed,
            "total_chunks": count,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "embed_model": self.embed_model,
            "collection_name": self.collection_name,
        }


_retriever_instance: Optional[MedicalKnowladgeRetriever] = None
    
def get_retriever(reset_cache: bool = False) -> MedicalKnowladgeRetriever:
    global _retriever_instance
    if _retriever_instance is None or reset_cache:
        _retriever_instance = MedicalKnowladgeRetriever(reset_cache=reset_cache)
    return _retriever_instance