# GPT-INDEX-BUILDER

This project is a versatile and powerful search tool that leverages state-of-the-art natural language processing models to provide relevant and contextually rich results. The primary goal of this project is to build a semantic search engine for textual content from various sources such as PDF files and Wikipedia pages.

The project utilizes the GPT-3.5-turbo model for generating responses and French Semantic model to create embeddings of textual data. Users can build an index of embeddings from a PDF file or a Wikipedia page, explore the index interactively, and deploy the search functionality on Telegram. The search results are presented as the top k relevant chunks of information, which are then used as context to generate an informative response from the GPT-3.5-turbo model.

The project is implemented in Python, and it employs several open-source libraries such as Click, OpenAI, Wikipedia, PyTorch, Tiktoken, and Rich. The code is organized into modular functions and classes, making it easy to understand, maintain, and extend. The main script provides a command-line interface for users to interact with the project's functionalities.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
   - [Build Index from PDF](#build-index-from-pdf)
   - [Build Index from Wikipedia](#build-index-from-wikipedia)
   - [Explore Index](#explore-index)
   - [Deploy on Telegram](#deploy-on-telegram)
3. [Requirements](#requirements)

## Installation

To install the necessary dependencies, run the following command:

```bash
python -m venv env 
source env/bin/activate
pip install --upgrade pip 
pip install -r requirements.txt
```

# Usage

## Supported Transformer Models

This project supports a variety of transformer models, including models from the Hugging Face Model Hub and sentence-transformers. Below are some examples:
    - Hugging Face Model: 'Sahajtomar/french_semantic'
    - Sentence-Transformers Model: 'paraphrase-MiniLM-L6-v2', 'all-mpnet-base-v2', etc...

Please ensure that the model you choose is compatible with the project requirements and adjust the `--transformer_model_name` option accordingly.


## Build Index from PDF
To build an index from a PDF file, run the following command:

```bash
export OPENAI_API_KEY=sk- TRANSFORMERS_CACHE=path2cache_folder; 
python main.py --transformer_model_name 'Sahajtomar/french_semantic' build-index-from-pdf 
    --path2pdf_file /path/to/file.pdf 
    --path2extracted_features /path/to/features.pkl
    --chunk_size 128
    --batch_size 8
```

## Build Index from Wikipedia
To build an index from a Wikipedia page, run the following command:

```bash
export OPENAI_API_KEY=sk- TRANSFORMERS_CACHE=path2cache_folder; 
python main.py --transformer_model_name 'Sahajtomar/french_semantic' build-index-from-wikipedia 
    --wikipedia_url https://url/to/wikipedia_page 
    --path2extracted_features /path/to/features.pkl
    --chunk_size 128
    --batch_size 8
```

# Explore Index
To explore the index, run the following command:

```bash
export OPENAI_API_KEY=sk- TRANSFORMERS_CACHE=path2cache_folder; 
python main.py --transformer_model_name 'Sahajtomar/french_semantic' explore-index
    --name service_name 
    --description service_description  
    --path2extracted_features /path/to/features.file 
    --top_k 11 
```

# Deploy on Telegram
To deploy the service on Telegram, run the following command:

```bash
export OPENAI_API_KEY=sk- TRANSFORMERS_CACHE=path2cache_folder; 
python main.py --transformer_model_name 'Sahajtomar/french_semantic' deploy-on-telegram 
    --name service_name 
    --description service_description  
    --path2extracted_features /path/to/features.file 
    --top_k 11 
    --telegram_token XXXX....XXXX
```