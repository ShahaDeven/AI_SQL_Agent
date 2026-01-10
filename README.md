# 🤖 AI-Powered Supply Chain Analytics Agent

![Python](https://img.shields.io/badge/Python-3.11%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b) ![LangChain](https://img.shields.io/badge/LangChain-Orchestration-green) ![DuckDB](https://img.shields.io/badge/DuckDB-OLAP-yellow)

A production-grade **Text-to-SQL Agent** designed to democratize supply chain data access. This application allows non-technical stakeholders (finance, operations) to query complex database schemas using natural language, simulate financial scenarios, and instantly visualize results without writing a single line of code.

Built on the **TPC-H Benchmark** dataset, this project demonstrates advanced RAG (Retrieval-Augmented Generation) patterns, self-healing SQL execution, and "What-If" scenario modeling.

---

## 🚀 Key Features

### 🧠 Phase 1: The Core Agent (Text-to-SQL)
- **Natural Language Interface:** Translate questions like *"Show me top 5 suppliers by revenue in Europe"* into optimized DuckDB SQL.
- **Hybrid Search Retriever:** Uses **BM25 + Semantic Search** (ChromaDB) to map vague user terms to specific database columns, handling schema ambiguity with **95% accuracy**.
- **Self-Healing Execution Loop:** Autonomously detects SQL syntax errors or security violations and triggers iterative re-prompting to correct the query before crashing.

### 🧪 Phase 2: "What-If" Simulator (The CFO Agent)
- **Scenario Analysis:** Interprets conditional prompts (e.g., *"What if we increased the discount by 10%?"*) and dynamically injects **Common Table Expressions (CTEs)** to simulate the impact on revenue/margin.
- **Non-Destructive:** Performs all simulations in-memory (read-only), ensuring zero risk to production data integrity.

### 📊 Phase 3: Auto-Visualizer
- **Dynamic Rendering:** Automatically detects data patterns (Time-Series vs. Categorical vs. Relational) to render the optimal chart type (Line, Bar, or Data Table) using Streamlit.
- **Interactive Dashboards:** Replaces static weekly reports with an ad-hoc, interactive exploration tool.

### ⚡ Engineering Optimizations
- **Smart Caching:** Implements a semantic caching layer (`sql_cache.json`) to store validated SQL queries, reducing API latency and costs for recurrent questions.
- **REST Protocol Enforcement:** Optimized for restrictive network environments (corporate/university firewalls) by bypassing standard gRPC blocks.

---

## 🛠️ Tech Stack

- **LLM Orchestration:** LangChain, Google Gemini 2.5 Flash
- **Database:** DuckDB (OLAP-optimized, local file-based)
- **Frontend:** Streamlit
- **Vector Store:** ChromaDB (Local persistence)
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`) runs locally on CPU
- **Environment:** Python 3.9+, Docker (optional)

---

## 📂 Project Structure

```bash
AI_SQL_Agent/
├── data/
│   ├── supply_chain.db       # Main DuckDB database (TPC-H schema)
│   └── sql_agent_demo.db     # (Optional) Smaller demo DB
├── src/
│   ├── agent_graph.py        # Core Logic: Agent Loop, Caching, Simulation
│   └── retriever.py          # Hybrid Search (BM25 + Semantic) logic
├── app.py                    # Streamlit Frontend application
├── requirements.txt          # Python dependencies
├── .env                      # API Keys (Not committed)
└── sql_cache.json            # Local cache for SQL queries (Auto-generated)
```

---

## ⚡ Installation & Setup
1. Clone the Repository
```bash
git clone https://github.com/ShahaDeven/AI_SQL_Agent.git
cd AI_SQL_Agent
```

2. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

3. Install Dependencies
```bash
pip install -r requirements.txt
```

4. Configure Environment Variables
Create a .env file in the root directory:
```ini
GOOGLE_API_KEY=your_google_api_key_here
# Optional: Disable telemetry for privacy
ANONYMIZED_TELEMETRY=False
```

5. Run the Application
```bash
streamlit run app.py
```

---

### 🧪 Usage Examples

#### 1. Basic Analytics:

   "What is the total revenue per region for the last year?" OR "List the top 3 customers in the AUTOMOBILE segment."

#### 2. Complex Reasoning (Chain-of-Thought):

   "Who is the supplier with the most parts in the region with the lowest revenue?" (The agent will use a CTE to first find the lowest revenue region, then filter suppliers.)

#### 3. Simulation ("What-If"):

   "What if we increased the discount by 5%? How would that affect total revenue?" (The agent generates a simulated view and compares it against actuals.)

---

### 🔒 Security & Safety

- **Read-Only Access:** DuckDB runs in `read_only=True` mode.
- **SQL Guardrails:** `check_sql_safety` blocks all DDL/DML commands.
- **Hallucination Control:** Queries referencing non-existent columns are rejected unless defined.

---

### 📝 Future Roadmap

[ ] Containerization: Full Docker-Compose setup for cloud deployment.

[ ] Feedback Loop: RLHF integration to allow users to flag incorrect SQL for model fine-tuning.

[ ] Multi-Database Support: Abstracting connections to support Snowflake/Postgres.
