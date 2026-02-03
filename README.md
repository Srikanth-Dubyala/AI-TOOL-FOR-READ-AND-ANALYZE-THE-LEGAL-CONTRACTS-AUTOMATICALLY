#####  AI-TOOL-FOR-READ-AND-ANALYZE-THE-LEGAL-CONTRACTS-AUTOMATICALLY
#### This project is an end-to-end AI platform that reads, analyzes, and evaluates contract documents using multiple domain-specialized agents powered by oLlama-3. The system automatically detects finance, legal, operational, and compliance risks and generates a structured, professional report.


#####  Multi-Agent Risk Analysis

Finance risk detection
Legal risk assessment
Operational vulnerability analysis
Compliance violations

#####  Document Ingestion
Supports PDF, DOCX, TXT
###### Text chunking via RecursiveCharacterTextSplitter
Cleans & normalizes malformed JSON from LLM outputs

###### AI Model
Meta Llama-3-8B-Instruct (HuggingFace Inference)
Sentence-Transformer embeddings

##### Vector Search Layer
Pinecone vector database for clause storage and semantic retrieval

###### Frontend Dashboard
Streamlit interface
Risk badges, domain-wise analysis, clause insights
Downloadable PDF report (ReportLab)
###### Deployment
Hosted on Streamlit Cloud
Fully browser-based, no setup required

###### Tech Stack
Python
LangChain, LangGraph
HuggingFace Inference
Pinecone
Streamlit
ReportLab
Sentence Transformers

###### Architecture
Upload contract →Text parsing & splitting →Parallel multi-agent LLM analysis →JSON cleaning & structuring →Embedding + storage in Pinecone →Frontend visualization →Exportable PDF report
