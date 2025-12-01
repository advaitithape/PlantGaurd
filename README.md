# 🌿 PlantGuard  
### A Multi-Agent System for Plant Disease Diagnosis, Treatment Guidance & Follow-Up  
**Capstone Project for Agentic AI Course**

**Repo link:** (add after pushing)

---

##  Overview
PlantGuard is an AI-powered multi-agent system that helps farmers diagnose plant diseases from images, receive safe treatment guidance grounded in a curated knowledge base, and get proactive follow-up support.

It combines:

- Computer vision  
- Retrieval-Augmented Generation (RAG)  
- Multi-agent orchestration  
- Long-term memory  
- Scheduled follow-ups  

Built using **Python**, **Gemini/OpenAI LLM**, and **Agent Development Kit (ADK)** patterns.

---

##  Why PlantGuard?

Traditional ML models stop at predicting a disease label. Farmers still struggle with:

- What treatment to apply?
- How much? How often?
- Organic or chemical?
- What if symptoms worsen?
- When to check again?

PlantGuard closes this gap by offering **end-to-end agricultural assistance**, not just classification.

---

##  Architecture Summary

### **Agents Included**
1. **Diagnosis Agent**
   - Validates image  
   - Performs disease classification  
   - Applies confidence-based safety gating  

2. **Knowledge Retrieval Agent**
   - Retrieves treatments from curated KB  
   - Supports FAISS semantic search  

3. **Advisory Agent (LLM)**
   - Generates safe, farmer-friendly instructions  
   - Ensures prescriptions come only from KB (“hallucination guard”)  
   - Produces JSON-structured responses  

4. **Follow-Up Agent**
   - Schedules future check-ins  
   - Uses long-term memory  
   - Generates follow-up messages via RAG  

5. **Orchestrator**
   - End-to-end pipeline combining classifier → RAG → follow-up  

---

