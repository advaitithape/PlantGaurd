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

# 🚀 **How to Test the Deployed API Using Postman**

This project exposes a REST API that allows you to classify plant leaf images and trigger follow-ups.
You can test everything easily using **Postman** — no code required.

---

## ✅ **1. Test Health Endpoint**

This confirms the API is running.

### **POSTMAN Request**

* **Method:** `GET`
* **URL:**

  ```
  https://plantgaurd.onrender.com/health
  ```

### **Expected Response**

```json
{
  "status": "ok"
}
```

---

## ✅ **2. Test Disease Classification (Main Pipeline)**

This is the core workflow — upload an image → get classification → RAG explanation → follow-up scheduled.

### **POSTMAN Request**

* **Method:** `POST`
* **URL:**

  ```
  https://plantgaurd.onrender.com/classify
  ```
* **Body:** `form-data`

  | Key               | Type   | Value                                             |
  | ----------------- | ------ | --------------------------------------------------|
  | **image**         | *file* | Upload the plant leaf image provided with the repo|
  | **user_id**       | text   | any string (e.g., `test_user`)                    |
  | **followup_days** | text   | optional (e.g., `0.002` for quick testing)        |

### Example:

* Set **Body → form-data**
* Add file:
  `image` → *(Select File)*
* Add text:
  `user_id` → `farmer123`
  `followup_days` → `0.002` (≈3 minutes)

Click **Send**.

### **Expected Response Structure**

```json
{
  "classification": { ... },
  "rag": { ... },
  "followup": {
    "id": "uuid",
    "status": "pending",
    "due_ts": 123456789
  }
}
```

This confirms:

* The model loaded successfully
* The RAG agent produced advice
* A follow-up has been scheduled

---

## ✅ **3. Test Follow-Up Trigger (Optional)**

You can force-trigger a follow-up event.

### **POSTMAN Request**

* **Method:** `POST`
* **URL:**

  ```
  https://plantgaurd.onrender.com/followup/trigger
  ```
* **Body:** `raw` → JSON

```json
{
  "followup_id": "PUT_ID_HERE"
}
```

You get the `followup_id` from the `/classify` response.

### **Expected Response**

```json
{
  "triggered": true
}
```

---

## 📌 Notes for Testers / Judges

* The system **does not render a webpage** — responses are JSON only.
* Image classification + RAG generation + follow-up scheduling all work via REST API.
* You can repeat `/classify` using different images to test accuracy and robustness.
* Follow-ups appear quickly if you set a small delay (e.g., `0.002` days ≈ 3 minutes).

---

