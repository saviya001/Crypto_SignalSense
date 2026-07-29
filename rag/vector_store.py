import os
import glob
import logging
from typing import List, Dict, Any

logging.getLogger("pypdf").setLevel(logging.ERROR)

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

class SimpleVectorStore:
    """
    Lightweight, robust RAG Vector Store implementation with PDF and Markdown ingest support.
    Supports Section 4 (d) requirements for domain corpus ingestion (20+ documents).
    """
    def __init__(self, corpus_dir: str = None):
        if corpus_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            corpus_dir = os.path.join(base_dir, "corpus")
        self.corpus_dir = corpus_dir
        self.documents: List[Dict[str, str]] = []
        self._load_corpus()

    def _load_corpus(self):
        """Loads all PDF and markdown files from the corpus directory."""
        if not os.path.exists(self.corpus_dir):
            return
        
        # Load Markdown files
        md_files = glob.glob(os.path.join(self.corpus_dir, "*.md"))
        for filepath in sorted(md_files):
            filename = os.path.basename(filepath)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    self.documents.append({
                        "filename": filename,
                        "title": filename.replace(".md", "").replace("_", " ").title(),
                        "content": content,
                        "format": "Markdown"
                    })
            except Exception as e:
                print(f"Error loading MD {filename}: {e}")

        # Load PDF files if available
        if HAS_PYPDF:
            pdf_files = glob.glob(os.path.join(self.corpus_dir, "*.pdf"))
            for filepath in sorted(pdf_files):
                filename = os.path.basename(filepath)
                try:
                    reader = PdfReader(filepath)
                    pdf_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                    if pdf_text.strip():
                        self.documents.append({
                            "filename": filename,
                            "title": filename.replace(".pdf", "").replace("_", " ").title(),
                            "content": pdf_text,
                            "format": "PDF"
                        })
                except Exception as e:
                    print(f"Error loading PDF {filename}: {e}")

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves the top_k relevant document chunks based on keyword relevance scoring."""
        if not self.documents:
            return []

        keywords = [kw.lower() for kw in query_text.replace("/", " ").replace("-", " ").split() if len(kw) > 2]
        scored_docs = []

        for doc in self.documents:
            score = 0
            doc_content_lower = doc["content"].lower()
            doc_title_lower = doc["title"].lower()

            for kw in keywords:
                if kw in doc_title_lower:
                    score += 5
                if kw in doc_content_lower:
                    score += doc_content_lower.count(kw)

            if score > 0:
                scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in scored_docs[:top_k]:
            results.append({
                "source": doc["filename"],
                "title": doc["title"],
                "content": doc["content"],
                "relevance_score": float(score),
                "format": doc.get("format", "Document")
            })

        if not results:
            for doc in self.documents[:top_k]:
                results.append({
                    "source": doc["filename"],
                    "title": doc["title"],
                    "content": doc["content"][:300] + "...",
                    "relevance_score": 1.0,
                    "format": doc.get("format", "Document")
                })

        return results

if __name__ == "__main__":
    store = SimpleVectorStore()
    print(f"Loaded {len(store.documents)} total corpus documents (PDFs + Markdown).")
    res = store.query("RSI overbought strategy")
    print(f"Query Result Count: {len(res)}")
    for r in res:
        print(f"- [{r['source']}] Format: {r['format']} | Score: {r['relevance_score']}")
