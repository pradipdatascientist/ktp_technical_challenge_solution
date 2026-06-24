import time
import json
from rag_pipeline import RAGSystem

def evaluate():
    rag = RAGSystem()
    test_queries = [
        "Find documents with many figures",
        "Which papers mention tables?",
        "Show me documents with titles",
        "Are there any lists in the data?"
    ]
    
    results = []
    for query in test_queries:
        print(f"Evaluating query: {query}")
        
        # Baseline
        start = time.time()
        baseline_res = rag.retrieve_baseline(query)
        baseline_time = time.time() - start
        
        # Enhanced
        start = time.time()
        enhanced_res = rag.retrieve_enhanced(query)
        enhanced_time = time.time() - start
        
        results.append({
            "query": query,
            "baseline": {
                "time": baseline_time,
                "num_results": len(baseline_res)
            },
            "enhanced": {
                "time": enhanced_time,
                "num_text_results": len(enhanced_res["text_results"]),
                "num_image_results": len(enhanced_res["image_results"]),
                "num_kg_facts": len(enhanced_res["kg_facts"])
            }
        })
        
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("Evaluation complete. Results saved to evaluation_results.json")

if __name__ == "__main__":
    evaluate()
