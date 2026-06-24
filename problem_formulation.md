# Problem Formulation

## 1. Overview

The primary goal of this technical challenge is to evaluate the ability to build practical AI components relevant to Retrieval-Augmented Generation (RAG). Specifically, the focus is on developing a system that can answer natural language queries about scientific content by leveraging multimodal data, knowledge graphs, reasoning, and explainability.

## 2. Core Problem

Traditional RAG systems primarily rely on textual information. However, scientific documents often contain rich multimodal data, including figures, tables, and complex layouts, which are crucial for comprehensive understanding and accurate information retrieval. The core problem is to effectively integrate these non-textual modalities and structured knowledge into a RAG pipeline to improve retrieval quality, reasoning, and explainability.

## 3. Sub-problems

To address the core problem, the following sub-problems need to be solved:

### 3.1. Data Ingestion, Preprocessing, and Multimodal Representation

*   **Challenge:** Scientific documents are complex, containing text, images, and structured metadata. Efficiently ingesting and preprocessing this diverse data, and creating a unified multimodal representation, is crucial.
*   **Solution Approach:** Utilize the PubLayNet dataset, which provides document images, extracted text, and region annotations. Develop a pipeline to parse this data, extract relevant features from both text and images, and prepare them for downstream tasks.

### 3.2. Knowledge Graph Construction or Integration

*   **Challenge:** Extracting structured knowledge (entities, relations) from unstructured and semi-structured scientific text and integrating it into a knowledge graph can enhance reasoning. The challenge lies in accurately identifying and linking these entities and relations.
*   **Solution Approach:** Implement techniques for entity and relation extraction from the preprocessed text and metadata. Construct a knowledge graph that captures key information like authors, figures, tables, and their relationships within the documents.

### 3.3. Retrieval and Generation (RAG-style)

*   **Challenge:** Develop a RAG pipeline that can effectively retrieve relevant information from both textual and multimodal sources, and then generate coherent and accurate answers. This involves designing efficient indexing and retrieval mechanisms for diverse data types.
*   **Solution Approach:** Implement two RAG approaches: a baseline text-only RAG and an enhanced multimodal RAG. The enhanced approach will leverage multimodal embeddings (e.g., CLIP) and knowledge graph queries to enrich the retrieval process.

### 3.4. Basic Reasoning and Answer Generation

*   **Challenge:** The system should not just retrieve information but also perform basic reasoning to synthesize answers. This requires the generation component to effectively utilize the retrieved context.
*   **Solution Approach:** Employ a Large Language Model (LLM) for answer generation, conditioned on the retrieved context. Prompt engineering will be critical to guide the LLM in producing relevant and reasoned responses.

### 3.5. Explainability of the Outputs

*   **Challenge:** For scientific applications, understanding *why* a particular answer was generated is as important as the answer itself. Ensuring explainability, especially for the enhanced multimodal approach, is a key requirement.
*   **Solution Approach:** Design the RAG pipeline to allow for tracing the sources of information (text chunks, images, KG facts) that contributed to the generated answer. The answer generation process should ideally highlight these contributing factors.

## 4. Comparison of Approaches

The task requires a quantitative comparison between two approaches:

*   **Baseline RAG:** A text-only RAG system using embeddings from document text/chunks.
*   **Enhanced Approach:** A multimodal RAG system incorporating visual features (e.g., CLIP, ResNet, layout information) and a knowledge graph (constructed from entities/relations in documents or linked to an external KG).

This comparison will highlight the benefits of integrating multimodal and structured knowledge for scientific content understanding.
