#!/bin/bash
# setup.sh: Install dependencies and setup environment for RAG CLI
set -e

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch==2.3.1 transformers==4.41.0 sentence-transformers==4.1.0 haystack-ai==2.6.1 click==8.1.7 numpy==1.26.4 psutil==6.0.0 fuzzywuzzy==0.18.0 python-Levenshtein==0.25.1

echo "Setup complete. Activate environment with: source venv/bin/activate"
echo "Run the CLI with: python rag_cli.py"
echo "Run evaluation with: python rag_cli.py --evaluate"