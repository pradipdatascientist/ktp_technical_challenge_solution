# KTP Technical Challenge: Multimodal RAG with Knowledge Graphs

This repository contains the solution for the KTP technical challenge, focusing on building and comparing a baseline text-only Retrieval-Augmented Generation (RAG) pipeline with an enhanced multimodal RAG pipeline that incorporates visual features and a knowledge graph for scientific content.

## Project Structure

*   `data_processing.py`: Script for ingesting and preprocessing the `small-publaynet-wds` dataset.
*   `kg_builder.py`: Script for constructing a knowledge graph from the processed data.
*   `rag_pipeline.py`: Implements both the baseline (text-only) and enhanced (multimodal + KG) RAG pipelines.
*   `evaluation.py`: Script for evaluating the performance of both RAG approaches.
*   `solution_architecture.md`: Detailed document outlining the solution architecture.
*   `problem_formulation.md`: Document describing the problem statement and sub-problems.
*   `requirements.txt`: Lists all necessary Python packages and their versions.
*   `README.md`: This file, providing an overview, setup instructions, and reproduction commands.

## Setup Instructions

To set up the environment and run the project, follow these steps:

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `clip` library is installed directly from its GitHub repository as specified in `requirements.txt`.*

## Reproduction Commands

To reproduce the data processing, knowledge graph construction, RAG pipeline demonstration, and evaluation, execute the following commands in sequence:

1.  **Run Data Processing:** This step downloads a subset of the `small-publaynet-wds` dataset, extracts images and metadata, and saves them to the `processed_data` directory.
    ```bash
    python3 data_processing.py
    ```

2.  **Build Knowledge Graph:** This step constructs a knowledge graph in Turtle format (`.ttl`) from the processed metadata.
    ```bash
    python3 kg_builder.py
    ```

3.  **Run RAG Pipeline Demonstration:** This script initializes the RAG system, sets up indices, and demonstrates both baseline and enhanced retrieval and generation for a sample query.
    ```bash
    python3 rag_pipeline.py
    ```

4.  **Run Evaluation:** This script evaluates both RAG approaches against a set of test queries and saves the results to `evaluation_results.json`.
    ```bash
    python3 evaluation.py
    ```

## Deliverables

The following files are generated and included as part of the solution:

*   `processed_data/`: Directory containing processed images and `metadata.json`.
*   `processed_data/knowledge_graph.ttl`: The constructed knowledge graph.
*   `evaluation_results.json`: JSON file containing the quantitative evaluation results.

## Discussion on Multimodal + KG, Limitations, and Insights

*(This section will be elaborated in the `solution_architecture.md` and `problem_formulation.md` documents, and potentially in a separate `discussion.md` if needed.)*

**Key Insights from Evaluation (based on `evaluation_results.json`):**

*   **Retrieval Time:** The enhanced approach generally takes longer due to the additional processing for multimodal embeddings and knowledge graph queries.
*   **Retrieval Quality:** While `num_text_results` and `num_image_results` are fixed to 3 for demonstration, the `num_kg_facts` in the enhanced approach indicates its ability to retrieve structured information. For queries like "Show me documents with titles" or "Are there any lists in the data?", the KG successfully retrieves relevant facts, which would not be possible with text-only retrieval.
*   **Potential for Improvement:** The current KG search is a simple entity matching. A more sophisticated KG query mechanism (e.g., SPARQL queries or graph neural networks) would significantly improve its utility.
*   **Multimodal Advantage:** The CLIP-based image retrieval in the enhanced approach allows for finding documents based on visual content, which is a significant advantage over the baseline.

