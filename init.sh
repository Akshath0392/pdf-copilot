#!/bin/bash

# LLM LIBRARIES
pip3 install tiktoken
pip3 install transformers
pip3 install -U "pydantic>=2.11,<3" "pydantic-settings>=2.4,<3"
pip3 install -U "langchain>=0.2,<0.4" "langchain-core>=0.2,<0.4" \
               "langchain-community>=0.2,<0.4" "langchain-pinecone>=0.2.8"
pip3 install langchain-anthropic
pip3 install langchain-openai
pip3 install langchain-google-genai
pip3 install langchain-huggingface
pip3 install sentence-transformers
pip3 install langchain-classic

# VECTOR DATABASES
pip3 install faiss-cpu==1.10.0
pip3 install pinecone

# AUXILIAR -> Deal with PDFs and get data online
pip3 install wikipedia
pip3 install pypdf

# BASICS
pip3 install pandas
pip3 install matplotlib
pip3 install python-dotenv
