import unittest
from rag.vector_store import SimpleVectorStore

class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        self.store = SimpleVectorStore()

    def test_corpus_loading(self):
        self.assertGreaterEqual(len(self.store.documents), 20)

    def test_query_retrieval(self):
        res = self.store.query("RSI overbought", top_k=2)
        self.assertGreater(len(res), 0)
        self.assertIn("source", res[0])

if __name__ == "__main__":
    unittest.main()
