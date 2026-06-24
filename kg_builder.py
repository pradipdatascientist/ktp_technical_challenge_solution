import json
import os
from rdflib import Graph, Literal, RDF, URIRef, Namespace

def build_kg(data_path="processed_data/metadata.json", output_path="processed_data/knowledge_graph.ttl"):
    with open(data_path, "r") as f:
        data = json.load(f)
        
    g = Graph()
    EX = Namespace("http://example.org/publaynet/")
    g.bind("ex", EX)
    
    for doc in data:
        doc_uri = URIRef(EX[doc["id"]])
        g.add((doc_uri, RDF.type, EX.Document))
        g.add((doc_uri, EX.hasText, Literal(doc["text"])))
        
        for i, region in enumerate(doc["regions"]):
            region_uri = URIRef(EX[f"{doc['id']}_region_{i}"])
            g.add((region_uri, RDF.type, EX[region["category"].capitalize()]))
            g.add((doc_uri, EX.contains, region_uri))
            
            if region["bbox"]:
                g.add((region_uri, EX.hasBBox, Literal(str(region["bbox"]))))

    g.serialize(destination=output_path, format="turtle")
    print(f"Knowledge Graph built and saved to {output_path}")

if __name__ == "__main__":
    build_kg()
