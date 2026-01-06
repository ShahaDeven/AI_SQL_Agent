# 🤖 AI Supply Chain Agent (RAG + SQL)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/Orchestration-LangChain-orange?logo=langchain)
![Gemini](https://img.shields.io/badge/AI-Gemini%202.5%20Flash-4285F4?logo=google&logoColor=white)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![DuckDB](https://img.shields.io/badge/Database-DuckDB-FFF000?logo=duckdb&logoColor=black)

An intelligent **SQL Agent** capable of answering complex business questions about Supply Chain data (TPC-H).

Unlike basic "Text-to-SQL" bots, this agent uses **Retrieval Augmented Generation (RAG)** to understand custom business logic (e.g., *"High Risk"* customers) and employs **Self-Healing** mechanisms to correct its own SQL errors.

---

## 🚀 Key Features

| Feature | Description |
| :--- | :--- |
| **🧠 Semantic Understanding** | Uses a Vector Database (**ChromaDB**) to inject business context. It knows that *"Revenue"* is calculated as `total_value * (1 - promo_reduction)`, logic that exists nowhere in the schema itself. |
| **🛡️ Enterprise Security** | • **Database:** Strict `READ_ONLY` connection to DuckDB.<br>• **App Level:** `sqlparse` logic blocks destructive commands (`DROP`, `DELETE`, `UPDATE`) before execution. |
| **❤️‍🩹 Self-Healing SQL** | If a generated query fails (syntax or logic), the agent reads the error, reflects, and **retries automatically** up to 3 times. |
| **📊 Dynamic UI** | Built with **Streamlit**, featuring interactive data tables and a live Database Schema Explorer. |
| **⚡ Dual-Mode Engine** | Automatically switches between a full **1GB local dataset** and a lightweight **"Demo Mode" (100MB)** for cloud deployment. |

---

## 🏗️ Architecture

The system follows a multi-step reasoning pipeline to ensure accuracy and safety.

```mermaid
graph TD
    User[👤 User Query] --> UI[💻 Streamlit UI]
    UI --> Agent[🤖 LangChain Agent]
    
    subgraph "Knowledge & Logic"
        Agent -- "1. Retrieve Context" --> RAG[🔍 ChromaDB Vector Store]
        RAG -- "Definitions & Logic" --> Agent
        Agent -- "2. Generate SQL" --> LLM[🧠 Gemini 2.5 Flash]
        LLM -- "Raw SQL" --> Agent
    end
    
    subgraph "Safety & Execution"
        Agent -- "3. Sanitize" --> Guard[🛡️ SQL Security Check]
        Guard -- "Safe SQL" --> DB[(🦆 DuckDB)]
        DB -- "Data / Error" --> Agent
    end

    Agent -- "4. Self-Correction Loop" --> LLM
    Agent --> Final[📊 Final Response]
```
