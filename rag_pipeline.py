import os
import json
import torch
import clip
from PIL import Image
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rdflib import Graph, Namespace

class RAGSystem:
    def __init__(self, data_path="processed_data/metadata.json", kg_path="processed_data/knowledge_graph.ttl"):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        
        self.kg = Graph()
        self.kg.parse(kg_path, format="turtle")
        self.EX = Namespace("http://example.org/publaynet/")
        
        # Models
        print("Loading models...")
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cpu")
        
        self.setup_indices()

    def setup_indices(self):
        print("Setting up indices...")
        # Baseline: Text-only index
        texts = [d["text"] for d in self.data]
        text_embeddings = self.text_model.encode(texts)
        self.text_index = faiss.IndexFlatL2(text_embeddings.shape[1])
        self.text_index.add(text_embeddings.astype('float32'))
        
        # Enhanced: Multimodal (CLIP) index
        image_embeddings = []
        for d in self.data:
            image = self.preprocess(Image.open(d["image_path"])).unsqueeze(0)
            with torch.no_grad():
                image_feat = self.clip_model.encode_image(image)
            image_embeddings.append(image_feat.numpy().flatten())
        
        image_embeddings = np.array(image_embeddings)
        self.image_index = faiss.IndexFlatL2(image_embeddings.shape[1])
        self.image_index.add(image_embeddings.astype('float32'))

    def retrieve_baseline(self, query, k=3):
        query_vec = self.text_model.encode([query]).astype('float32')
        D, I = self.text_index.search(query_vec, k)
        results = [self.data[i] for i in I[0]]
        return results

    def retrieve_enhanced(self, query, k=3):
        # 1. Text Search
        query_vec_text = self.text_model.encode([query]).astype('float32')
        D_t, I_t = self.text_index.search(query_vec_text, k)
        
        # 2. Image Search (CLIP)
        with torch.no_grad():
            query_vec_clip = self.clip_model.encode_text(clip.tokenize([query])).numpy().astype('float32')
        D_i, I_i = self.image_index.search(query_vec_clip, k)
        
        # 3. KG Search (Simple entity matching)
        # In a real system, we'd use NER to find entities in the query
        kg_facts = []
        for doc in self.data:
            if any(word.lower() in doc["text"].lower() for word in query.split()):
                doc_uri = self.EX[doc["id"]]
                for s, p, o in self.kg.triples((doc_uri, None, None)):
                    kg_facts.append(f"{s} {p} {o}")
        
        combined_results = {
            "text_results": [self.data[i] for i in I_t[0]],
            "image_results": [self.data[i] for i in I_i[0]],
            "kg_facts": kg_facts[:5]
        }
        return combined_results

    def generate_answer(self, query, context, mode="baseline"):
        # This would normally call an LLM. Here we simulate the response.
        if mode == "baseline":
            context_str = "\n".join([c["text"] for c in context])
            answer = f"Based on the text-only retrieval, the answer to '{query}' is related to: {context_str[:100]}..."
        else:
            context_str = "\n".join([c["text"] for c in context["text_results"]])
            kg_str = "\n".join(context["kg_facts"])
            answer = f"Based on multimodal and KG retrieval, the answer to '{query}' incorporates visual evidence and structured facts: {kg_str[:100]}..."
        
        return answer

def run_demo():
    rag = RAGSystem()
    query = "Find documents with figures and tables"
    
    print(f"\nQuery: {query}")
    
    print("\n--- Baseline Approach ---")
    baseline_results = rag.retrieve_baseline(query)
    baseline_answer = rag.generate_answer(query, baseline_results, mode="baseline")
    print(f"Answer: {baseline_answer}")
    
    print("\n--- Enhanced Approach ---")
    enhanced_results = rag.retrieve_enhanced(query)
    enhanced_answer = rag.generate_answer(query, enhanced_results, mode="enhanced")
    print(f"Answer: {enhanced_answer}")
    print(f"Retrieved KG Facts: {len(enhanced_results['kg_facts'])}")

if __name__ == "__main__":
    run_demo()
