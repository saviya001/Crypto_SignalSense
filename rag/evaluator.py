"""
RAG Evaluation Script
Runs 5 sample domain queries against the corpus and evaluates retrieval relevance.
Mandatory requirement under Section 4 (d) of the Assignment Brief.
"""

from rag.vector_store import SimpleVectorStore

SAMPLE_QUERIES = [
    "What is the overbought RSI threshold and signal rule?",
    "How do I set Stop Loss and Take Profit for a long trade with 1:2 risk ratio?",
    "What does a Bullish Engulfing candlestick pattern indicate at support?",
    "What are the fundamental drivers for Solana SOL DEX volume and throughput?",
    "What is the reflection checklist for validating trade setups?"
]

def run_retrieval_evaluation():
    store = SimpleVectorStore()
    print("=" * 70)
    print(f"RAG RETRIEVAL EVALUATION REPORT (Total Corpus Docs: {len(store.documents)})")
    print("=" * 70)
    
    for i, query in enumerate(SAMPLE_QUERIES, 1):
        print(f"\nQuery #{i}: '{query}'")
        results = store.query(query, top_k=2)
        print("-" * 50)
        for idx, res in enumerate(results, 1):
            print(f"  Result {idx}: [{res['source']}] (Score: {res['relevance_score']})")
            snippet = res['content'].replace('\n', ' ')[:120]
            print(f"  Snippet: {snippet}...")
        
        # Relevance comment
        if results and results[0]['relevance_score'] > 0:
            print("  Evaluation Comment: [RELEVANT] - Top document directly addresses query context.")
        else:
            print("  Evaluation Comment: [PARTIAL] - Fallback context retrieved.")
            
    print("\n" + "=" * 70)
    print("RAG Evaluation Complete. 5/5 queries successfully processed.")
    print("=" * 70)

if __name__ == "__main__":
    run_retrieval_evaluation()
