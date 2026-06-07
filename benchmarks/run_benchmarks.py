import time
import torch
import json
from rag_engine import retrieve, generate, emb_model

# dummy evaluation questions representing different lengths
test_questions = [
    "what are the core definitions of a NP Hard Problem?",
    "explain the examples of NP Complete Problems in the document.",
    "How are all NP Complete problems also NP Hard?"
]

def run_matrix():
    results = []
    
    # matrix test 1: embedding placement variation
    for device_setting in ['cpu', 'cuda']:
        print(f"\n[matrix] configuring embedding model device to: {device_setting}")
        try:
            emb_model.to(device_setting)
        except Exception as e:
            print(f"could not shift embedding to {device_setting}: {e}")
            continue
            
        # matrix test 2: experimenting with retrieved context length (top-k chunks)
        for k_chunks in [3, 6]:
            print(f"[matrix] configuring faiss retrieval to return k = {k_chunks}")
            
            for q in test_questions:
                print(f" -> evaluating question: '{q[:40]}...'")
                
                # trigger manual retrieval with adjusted k-bounds
                start_retrieval = time.perf_counter()
                
                try:
                    context = retrieve(q, k=k_chunks)
                    retrieval_time = (time.perf_counter() - start_retrieval) * 1000
                    
                    # capture generation metrics
                    # we will read the terminal outputs or modify generate() to return metrics dictionary
                    ans, tokens, metrics = generate(context, q)
                    
                    # append the configuration state to trace later
                    results.append({
                        "embedding_device": device_setting,
                        "k_chunks": k_chunks,
                        "query": q,
                        "retrieval_time_ms": retrieval_time,
                        **metrics
                    })
                except Exception as e:
                    print(f"error during matrix loop execution: {e}")

    # save raw log state
    with open("benchmarks/matrix_raw_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== matrix run execution complete ===")

if __name__ == "__main__":
    run_matrix()