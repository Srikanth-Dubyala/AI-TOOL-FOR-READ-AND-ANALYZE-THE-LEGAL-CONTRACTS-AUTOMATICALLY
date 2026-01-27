#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import os
from typing import List, TypedDict

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.utils.json import parse_json_markdown

from langgraph.graph import StateGraph, END

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=3072,
    temperature=0.1,
    huggingfacehub_api_token=HUGGINGFACE_API_KEY
    
)

chat_model = ChatHuggingFace(llm=llm)


# In[4]:


from typing_extensions import Annotated
import operator

class AgentState(TypedDict):
    chunks: List
    analysis: Annotated[dict, operator.or_]



# In[5]:


def doc_types_and_split(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError("Unsupported file type")

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


# In[6]:



# In[21]:


DOMAIN_PROMPTS = {
   "finance": """
You are a FINANCE contract risk analyzer.

Analyze ONLY financial risks:
- payments
- penalties
- pricing
- billing terms
- financial liabilities
- compensation clauses

DO NOT analyze legal, operational, or compliance topics.

Rules:
- Return ONLY valid JSON
- No explanations
- No markdown
- No extra text
- If no financial risks are found, return an empty list
- Every object must contain all fields

JSON FORMAT:

{
  "finance": [
    {
      "clause": "exact contract clause text",
      "risk_level": "LOW|MEDIUM|HIGH",
      "reason": "why this is a risk",
      "impact": "financial impact",
      "recommendation": "how to reduce risk"
    }
  ]
}

Contract:
{contract}
"""
,

   "legal": """
You are a LEGAL contract risk analyzer.

Analyze ONLY legal risks:
- jurisdiction
- termination
- indemnity
- dispute resolution
- governing law
- liability clauses

DO NOT analyze finance, operations, or compliance topics.

Rules:
- Return ONLY valid JSON
- No explanations
- No markdown
- No extra text
- If no legal risks are found, return an empty list
- Every object must contain all fields

JSON FORMAT:

{
  "legal": [
    {
      "clause": "exact contract clause text",
      "risk_level": "LOW|MEDIUM|HIGH",
      "reason": "why this is a legal risk",
      "issue": "legal issue type",
      "explanation": "short explanation",
      "recommendation": "how to reduce risk"
    }
  ]
}

Contract:
{contract}
"""
,

  "operations": """
You are an OPERATIONS contract risk analyzer.

Analyze ONLY operational risks:
- delivery timelines
- execution issues
- dependencies
- performance obligations
- service availability

DO NOT analyze finance, legal, or compliance topics.

Rules:
- Return ONLY valid JSON
- No explanations
- No markdown
- No extra text
- If no operational risks are found, return an empty list
- Every object must contain all fields

JSON FORMAT:

{
  "operations": [
    {
      "clause": "exact contract clause text",
      "risk_level": "LOW|MEDIUM|HIGH",
      "reason": "why this is a risk",
      "impact": "operational impact",
      "action": "recommended action"
    }
  ]
}

Contract:
{contract}
"""
,

   "compliance": """
You are a COMPLIANCE contract risk analyzer.

Analyze ONLY compliance risks:
- regulations
- legal standards
- certifications
- statutory violations
- industry compliance

DO NOT analyze finance, legal, or operational topics.

Rules:
- Return ONLY valid JSON
- No explanations
- No markdown
- No extra text
- If no compliance risks are found, return an empty list
- Every object must contain all fields

JSON FORMAT:

{
  "compliance": [
    {
      "clause": "exact contract clause text",
      "risk_level": "LOW|MEDIUM|HIGH",
      "reason": "why this violates compliance",
      "violation": "specific violation",
      "required_action": "what must be done"
    }
  ]
}

Contract:
{contract}
"""

}

    
import json
import re


def safe_json_parse(text: str):
    import json, re
    text = re.sub(r"```json|```", "", text)
    
    # Fix bad backslashes that aren't followed by JSON-legal escapes
    text = re.sub(r'\\(?!["\\/bfnrt])', r"\\\\", text)
    # Replace double backslashes
    text = re.sub(r'\\\\', " ", text)

    text = text.strip()
    

    # If it doesn't even end like JSON, it's truncated
    if not text.endswith("}"):
        print("MODEL OUTPUT WAS TRUNCATED")
        print(text[-500:])
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fix bad escapes
        fixed = text.replace("\\", "\\\\")
        try:
            return json.loads(fixed)
        except Exception:
            print(" MODEL RETURNED INVALID JSON:")
            print(text[:1000])
            return None


# In[33]:


def analyzer_node(state: AgentState, domain: str):
    """
    Analyze a single domain using the LLM.
    """
    full_text = "\n".join(c.page_content for c in state.get("chunks", []))[:4000]

    prompt = DOMAIN_PROMPTS[domain].replace("{contract}", full_text)
    response = chat_model.invoke(prompt).content
    parsed = safe_json_parse(response)
    if parsed is None:
        print(f"{domain} agent returned invalid JSON")
        return []
    return parsed.get(domain, [])

def finance_node(state: AgentState):
    return {"analysis": {"finance": analyzer_node(state, "finance")}}

def legal_node(state: AgentState):
    return {"analysis": {"legal": analyzer_node(state, "legal")}}

def operations_node(state: AgentState):
    return {"analysis": {"operations": analyzer_node(state, "operations")}}

def compliance_node(state: AgentState):
    return {"analysis": {"compliance": analyzer_node(state, "compliance")}}


def join_node(state: AgentState):
    merged = {}
    for k, v in state.get("analysis", {}).items():
        merged[k] = v
    return {"analysis": merged}

def build_multi_agent_graph():
    graph = StateGraph(AgentState)

    # Start node
    graph.add_node("start", lambda state: state)

    # Parallel agent nodes
    graph.add_node("finance", finance_node)
    graph.add_node("legal", legal_node)
    graph.add_node("operations", operations_node)
    graph.add_node("compliance", compliance_node)

    # Join node
    graph.add_node("join", join_node)

    # Entry
    graph.set_entry_point("start")

    # FAN-OUT (parallel)
    graph.add_edge("start", "finance")
    graph.add_edge("start", "legal")
    graph.add_edge("start", "operations")
    graph.add_edge("start", "compliance")

    # FAN-IN (join)
    graph.add_edge("finance", "join")
    graph.add_edge("legal", "join")
    graph.add_edge("operations", "join")
    graph.add_edge("compliance", "join")

    # End
    graph.add_edge("join", END)

    return graph.compile()










# In[35]:


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
pc= Pinecone(api_key = os.environ.get("PINECONE_API_KEY"))
index_name = pc.Index("contract-analysis")
vectorstore = PineconeVectorStore(
    index=index_name,
    embedding=embedding_model
    )
print("vector initlaized successfully")



# In[36]:


def store_analysis_to_pinecone(analysis: dict, contract_id: str):
    texts = []
    metadatas = []
    ids = []

    def add(domain, items):
        for i, item in enumerate(items):
            clause = item.get("clause", "")
            if not clause.strip():
                continue
            texts.append(clause)
            metadatas.append({
                "contract_id": contract_id,
                "domain": domain,
                "risk_level": item.get("risk_level", "UNKNOWN"),
                "data": json.dumps(item)
            })
            ids.append(f"{contract_id}_{domain}_{i}")

    for domain in ["finance", "legal", "operations", "compliance"]:
        add(domain, analysis.get(domain, []))

    if texts:
        vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        print(f"Stored {len(texts)} vectors in Pinecone.")
    else:
        print("Warning: No vectors generated. Model returned empty sections.")





# In[37]:


import uuid

def generate_contract_id():
    return str(uuid.uuid4())


# In[40]:


def format_report_for_ui(analysis: dict):
    risk_order = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}

    def max_risk(items):
        if not items:
            return "LOW"
        return max([i.get("risk_level", "LOW") for i in items], key=lambda x: risk_order.get(x, 1))

    # Overview
    overview = {
        "overall_risk": max_risk([item for domain in analysis.values() for item in domain]),
        "summary": "Contract analysis generated by AI.",
        "finance_risk": max_risk(analysis.get("finance", [])),
        "legal_risk": max_risk(analysis.get("legal", [])),
        "operations_risk": max_risk(analysis.get("operations", [])),
        "compliance_risk": max_risk(analysis.get("compliance", [])),
    }

    def fill_items(domain_name, domain_items):
        for i in domain_items:
            if domain_name == "finance":
                i.setdefault("impact", i.get("impact", "N/A"))
                i.setdefault("recommendation", i.get("recommendation", "N/A"))
            elif domain_name == "legal":
                i.setdefault("issue", i.get("issue", "N/A"))
                i.setdefault("explanation", i.get("explanation", "N/A"))
                i.setdefault("recommendation", i.get("recommendation", "N/A"))
            elif domain_name == "operations":
                i.setdefault("type", i.get("type", "N/A"))
                i.setdefault("impact", i.get("impact", "N/A"))
                i.setdefault("action", i.get("action", "N/A"))
            elif domain_name == "compliance":
                i.setdefault("area", i.get("area", "N/A"))
                i.setdefault("violation", i.get("violation", "N/A"))
                i.setdefault("required_action", i.get("required_action", "N/A"))
        return domain_items

    return {
        "overview": overview,
        "finance": fill_items("finance", analysis.get("finance", [])),
        "legal": fill_items("legal", analysis.get("legal", [])),
        "operations": fill_items("operations", analysis.get("operations", [])),
        "compliance": fill_items("compliance", analysis.get("compliance", []))
    }

def run_contract_analysis(file_path: str):
    import uuid

    contract_id = str(uuid.uuid4())
    chunks = doc_types_and_split(file_path)
    app = build_multi_agent_graph()
    result = app.invoke({"chunks": chunks, "analysis": {}})
    analysis = result.get("analysis", {})
    store_analysis_to_pinecone(analysis, contract_id)
    report = format_report_for_ui(analysis)
    report["contract_id"] = contract_id
    return report



# In[38]:




# In[ ]:


























