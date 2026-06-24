# Solution Architecture Design

## 1. Introduction

This document outlines the proposed solution architecture for the KTP technical challenge, which involves building a Retrieval-Augmented Generation (RAG) pipeline to answer natural language queries about scientific content. The challenge requires comparing a baseline text-only RAG approach with an enhanced multimodal RAG approach that incorporates visual features and a knowledge graph.

## 2. Data Ingestion and Preprocessing

**Dataset:** The `small-publaynet-wds` dataset will be used, which contains document images, extracted text, figure/table regions, and associated metadata.

**Steps:**
1.  **Download and Extraction:** The dataset will be downloaded from Hugging Face. The webdataset format will be processed to extract individual documents, their text content, and associated images.
2.  **Text Processing (Baseline & Enhanced):**
    *   **Chunking:** Text will be split into smaller, overlapping chunks to facilitate retrieval. Standard NLP techniques for sentence splitting and tokenization will be applied.
    *   **Cleaning:** Basic text cleaning (e.g., removing special characters, extra whitespace) will be performed.
3.  **Image Processing (Enhanced):**
    *   **Feature Extraction:** Visual features will be extracted from document images. This may involve using pre-trained models like ResNet for general image features or CLIP for multimodal (image-text) embeddings.
    *   **Layout Analysis:** Information about the layout of text and figures on the page will be extracted from the metadata to understand the spatial relationships between different document elements.
    *   **Image Captioning (Optional but Recommended):** If not already present in the dataset, image captions could be generated using a dedicated image captioning model to provide textual descriptions of visual content.

## 3. Knowledge Graph Construction (Enhanced Approach)

**Purpose:** To represent structured knowledge about entities and their relationships within the scientific documents, enhancing retrieval and reasoning capabilities.

**Steps:**
1.  **Entity Extraction:** Identify key entities such as authors, institutions, figures, tables, sections, and results from the text and metadata. Named Entity Recognition (NER) models or rule-based patterns can be employed.
2.  **Relation Extraction:** Determine relationships between extracted entities (e.g., "Figure X describes Method Y", "Author A works at Institution B", "Table Z presents Results W"). This can be achieved using dependency parsing, pattern matching, or fine-tuned language models.
3.  **Graph Storage:** A graph database (e.g., Neo4j, or a simpler in-memory graph representation for this challenge) will store the entities as nodes and relationships as edges.

## 4. Embedding Generation

**Purpose:** To convert text, image, and knowledge graph components into numerical vector representations suitable for similarity search.

**Approaches:**
1.  **Text Embeddings (Baseline & Enhanced):**
    *   **Model:** Sentence Transformers (e.g., `all-MiniLM-L6-v2`) or OpenAI embeddings will be used to generate dense vector representations for text chunks and natural language queries.
2.  **Multimodal Embeddings (Enhanced):**
    *   **Model:** CLIP (Contrastive Language-Image Pre-training) will be used to generate aligned embeddings for both images and their corresponding textual descriptions (captions, surrounding text). This allows for direct comparison between image and text modalities.
    *   **Late Fusion (Alternative/Complementary):** Separate embeddings for text and visual features can be generated and then combined at a later stage (e.g., concatenation, weighted sum) before retrieval or re-ranking.
3.  **Knowledge Graph Embeddings (Enhanced):**
    *   **Model:** Techniques like TransE, ComplEx, or Graph Neural Networks (GNNs) can be used to embed entities and relations from the knowledge graph into a continuous vector space. This allows for semantic search within the graph.

## 5. Retrieval

**Purpose:** To efficiently find the most relevant pieces of information (text chunks, images, knowledge graph facts) given a natural language query.

**Approaches:**
1.  **Baseline RAG (Text-only):**
    *   **Vector Store:** A vector database (e.g., FAISS, ChromaDB, or a simple `scikit-learn` NearestNeighbors index for demonstration) will store the text embeddings.
    *   **Search:** Given a query, its embedding will be used to perform a similarity search (e.g., cosine similarity) to retrieve the top-k most relevant text chunks.
2.  **Enhanced RAG (Multimodal + KG):**
    *   **Hybrid Retrieval:** Combine multiple retrieval strategies:
        *   **Multimodal Vector Search:** Use the query embedding to search for relevant text chunks and image embeddings in their respective vector stores.
        *   **Knowledge Graph Traversal/Search:** Query the knowledge graph directly to retrieve structured facts related to entities mentioned in the natural language query. This could involve SPARQL-like queries or graph pattern matching.
        *   **Keyword Search:** (Optional) Use traditional keyword search for exact matches.
    *   **Re-ranking:** A re-ranking model (e.g., a cross-encoder or a simple heuristic-based re-ranker) will combine the results from different retrieval sources, considering factors like semantic similarity, relevance to the knowledge graph, and spatial proximity (from layout analysis).

## 6. Generation

**Purpose:** To synthesize a coherent and accurate answer based on the retrieved context and the natural language query.

**Approach (Baseline & Enhanced):**
1.  **Large Language Model (LLM):** A pre-trained LLM (e.g., a model from the OpenAI series, or an open-source alternative like Llama 2, Mistral) will be used. The retrieved context (text chunks, image captions, KG facts) will be provided to the LLM as part of the prompt.
2.  **Prompt Engineering:** Carefully crafted prompts will guide the LLM to generate concise, accurate, and explainable answers. For the enhanced approach, prompts will be designed to leverage the structured information from the knowledge graph and multimodal insights.

## 7. Evaluation

**Purpose:** To quantitatively compare the performance of the baseline and enhanced RAG approaches.

**Metrics:**
*   **Retrieval Metrics:** Precision, Recall, F1-score, Mean Reciprocal Rank (MRR), Normalized Discounted Cumulative Gain (NDCG) to assess the quality of retrieved documents/facts.
*   **Generation Metrics:** ROUGE, BLEU, METEOR for text generation quality. Human evaluation for factual accuracy, coherence, and relevance.
*   **Explainability:** Qualitative assessment of the model's ability to provide justifications for its answers, especially for the enhanced approach leveraging the knowledge graph.

## 8. Reproducibility

*   `requirements.txt` or `environment.yml` will list all necessary Python packages and their versions.
*   Setup instructions will detail how to set up the environment and download the dataset.
*   Exact commands will be provided to run the data processing, model training (if any), evaluation, and inference steps.
