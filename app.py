import os
import warnings
import logging

# ===== SUPPRESS ALL WARNINGS/TELEMETRY BEFORE OTHER IMPORTS =====
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CHROMA_TELEMETRY_IMPL"] = "posthog.Posthog"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["POSTHOG_DISABLED"] = "true"
os.environ["CHROMA_TELEMETRY"] = "False"

# Suppress logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('chromadb').setLevel(logging.ERROR)
logging.getLogger('chromadb.telemetry').setLevel(logging.CRITICAL)
logging.getLogger('posthog').setLevel(logging.CRITICAL)
logging.getLogger('httpx').setLevel(logging.WARNING)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', module='tensorflow')
warnings.filterwarnings('ignore', module='chromadb')
warnings.filterwarnings('ignore', message='.*torch.classes.*')

import streamlit as st
import duckdb
import pandas as pd
import time
from src.agent_graph import agent_workflow, DB_PATH
import src.agent_graph as agent_graph_module
from src.clarifier import check_needs_clarification, refine_question

st.set_page_config(page_title="AI SQL Agent", page_icon="🤖", layout="wide")

USER_AVATAR = "🧑‍💼"
BOT_AVATAR = "🤖"

@st.cache_resource
def load_agent_resources():
    print("⏳ Loading Heavy AI Models... (This should only happen once)")
    return agent_graph_module

agent_module = load_agent_resources()

def get_metrics():
    """Always get the current metrics object from the agent module."""
    return agent_module.metrics

DB_PATH = agent_module.DB_PATH 

def get_schema_for_ui():
    """
    Queries DuckDB to get a clean list of all tables and columns 
    so the user knows what they can ask about.
    """
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        tables = con.execute("SHOW TABLES").fetchdf()
        schema_data = {}

        for table_name in tables['name']:
            columns = con.execute(f"DESCRIBE {table_name}").fetchdf()
            schema_data[table_name] = columns[['column_name', 'column_type']]
            
        con.close()
        return schema_data
    except Exception as e:
        return {"Error": str(e)}

def auto_visualize(df):
    """
    Analyzes the DataFrame and renders the best Streamlit chart.
    """
    df.columns = df.columns.str.strip()

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()

    if any("year" in col.lower() or "date" in col.lower() for col in text_cols + numeric_cols):
        if numeric_cols:
            st.caption("📈 Detecting Trend Data... Switching to Line Chart")
            date_col = next(c for c in text_cols + numeric_cols if "year" in c.lower() or "date" in c.lower())
            st.line_chart(df.set_index(date_col)[numeric_cols])
            return

    if len(text_cols) == 1 and len(numeric_cols) > 0:
        st.caption("📊 Detecting Categorical Data... Switching to Bar Chart")
        st.bar_chart(df.set_index(text_cols[0])[numeric_cols])
        return

    st.dataframe(df, hide_index=True, use_container_width=True)


def execute_query(prompt: str):
    """Execute the query and handle the response."""
    try:
        response_data, sql_query = agent_module.agent_workflow(prompt)

        if isinstance(response_data, pd.DataFrame):
            if not response_data.empty:
                auto_visualize(response_data)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_data,
                    "type": "dataframe",
                    "sql": sql_query
                })
            else:
                st.warning("The query returned no results (Empty Table).")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "The query returned no results (Empty Table).",
                    "type": "text",
                    "sql": sql_query
                })

        elif response_data is None:
            error_msg = sql_query if sql_query else "I could not generate a valid answer."
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "type": "text"
            })

        elif isinstance(response_data, str):
            st.error(response_data)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_data,
                "type": "text",
                "sql": sql_query
            })
        else:
            st.error("I could not generate a valid answer.")

        if sql_query and not sql_query.startswith("MISSING") and not sql_query.startswith("Failed"):
            with st.expander("View Generated SQL"):
                st.code(sql_query, language="sql")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"An error occurred: {e}",
            "type": "text"
        })


# =============================================
# SIDEBAR
# =============================================
with st.sidebar:
    sidebar_tab = st.radio(
        "Navigation",
        ["📋 Schema", "📊 Metrics"], 
        horizontal=True, 
        key="sidebar_tab",
        label_visibility="collapsed"
    )

    if sidebar_tab == "📋 Schema":
        st.header("🗄️ Database Schema")
        st.markdown("Reference these tables when asking questions:")
        
        table_descriptions = {
            "customer": "👤 Registered users, account balances, and churn risk.",
            "orders": "📦 Order headers, dates, and priority status.",
            "lineitem": "🧾 Individual items in an order (price, discount, promo).",
            "nation": "🌐 Countries associated with customers and suppliers.",
            "region": "🗺️ Continents and geographic regions.",
            "part": "⚙️ Product catalog and specifications.",
            "supplier": "🏭 Companies that supply parts.",
            "partsupp": "🔗 Inventory linking parts to suppliers."
        }

        schema_info = get_schema_for_ui()
        
        if "Error" in schema_info:
            st.error(f"Could not load schema: {schema_info['Error']}")
        else:
            for table_name, df_columns in schema_info.items():
                desc = table_descriptions.get(table_name, "No description available.")
                with st.expander(f"**{table_name.upper()}**"):
                    st.caption(desc) 
                    st.dataframe(df_columns, hide_index=True, use_container_width=True)

    elif sidebar_tab == "📊 Metrics":
        st.header("📊 Agent Metrics")
        
        if st.button("🔄 Refresh Metrics", use_container_width=True):
            st.rerun()

        metrics = get_metrics()
        summary = metrics.get_summary()

        if summary["total_queries"] == 0:
            st.info("No queries yet. Ask a question to see live metrics!")
        else:
            col1, col2 = st.columns(2)
            col1.metric("Total Queries", summary["total_queries"])
            col2.metric("Success Rate", f"{summary['success_rate']}%")

            col3, col4 = st.columns(2)
            col3.metric("Cache Hit Rate", f"{summary['cache_hit_rate']}%")
            col4.metric("Avg Latency", f"{summary['avg_latency']}s")

            col5, col6 = st.columns(2)
            col5.metric("Est. Tokens Used", f"{summary['total_estimated_tokens']:,}")
            col6.metric("Self-Heals", summary["self_healing_recoveries"])

            st.divider()

            st.subheader("⏱️ Avg Latency Breakdown")
            breakdown = metrics.get_latency_breakdown()
            breakdown_df = pd.DataFrame({
                "Stage": ["Retriever", "LLM", "Database", "Other"],
                "Seconds": [
                    breakdown["retriever"],
                    breakdown["llm"],
                    breakdown["db"],
                    breakdown["other"]
                ]
            })
            st.bar_chart(breakdown_df.set_index("Stage"))

            st.divider()

            st.subheader("📝 Recent Queries")
            recent = metrics.get_recent_queries(5)
            for i, q in enumerate(reversed(recent)):
                status = "✅" if q["success"] else "❌"
                cache_tag = " ⚡cache" if q["cache_hit"] else ""
                sim_tag = " 🧪sim" if q["is_simulation"] else ""
                heal_tag = " 🔄healed" if q["self_healed"] else ""

                with st.expander(f"{status} {q['question'][:50]}...{cache_tag}{sim_tag}{heal_tag}"):
                    st.caption(f"Time: {q['timestamp']} | Latency: {q['total_latency']}s")
                    st.caption(f"LLM attempts: {q['llm_attempts']} | Rows: {q['rows_returned']}")
                    st.caption(f"Tokens (est): ~{q['estimated_prompt_tokens'] + q['estimated_completion_tokens']}")
                    if q["generated_sql"]:
                        st.code(q["generated_sql"], language="sql")
                    if q["error"]:
                        st.error(q["error"])


# =============================================
# MAIN CHAT AREA
# =============================================
st.title("🤖 AI SQL Agent")
st.markdown("Ask questions about your **Supply Chain Data** in plain English.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = None

if "original_question" not in st.session_state:
    st.session_state.original_question = None

# Display Chat History
for idx, message in enumerate(st.session_state.messages):
    if (st.session_state.pending_clarification and 
        message.get("type") == "clarification" and 
        idx == len(st.session_state.messages) - 1):
        continue
        
    icon = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=icon):
        if message.get("type") == "dataframe":
            auto_visualize(message["content"])
        elif message.get("type") == "clarification":
            st.markdown(message["content"])
        elif message.get("type") == "selection":
            st.markdown(f"**{message['content']}**")
        else:
            st.markdown(message["content"])

        if "sql" in message and message.get("sql"):
            with st.expander("View Generated SQL"):
                st.code(message["sql"], language="sql")

# =============================================
# HANDLE PENDING CLARIFICATION
# =============================================
if st.session_state.pending_clarification:
    clarification = st.session_state.pending_clarification
    
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        st.markdown(clarification.message)
        
        st.markdown("") 

        cols = st.columns(len(clarification.options))
        
        for i, option in enumerate(clarification.options):
            if cols[i].button(
                option, 
                key=f"clarify_{i}", 
                use_container_width=True
            ):
                original_q = st.session_state.original_question
                refined_q = refine_question(original_q, clarification.ambiguity_type, option)

                st.session_state.pending_clarification = None
                st.session_state.original_question = None
                
                st.session_state.messages.append({
                    "role": "user",
                    "content": option,
                    "type": "selection"
                })

                with st.spinner("🧠 Got it! Let me pull that data..."):
                    execute_query(refined_q)
                
                st.session_state["_last_query_completed"] = time.time()
                st.rerun()

        st.markdown("")
        st.caption("Or, if you'd prefer:")
        if st.button("↩️ Just try my original question as-is", key="skip_clarify"):
            original_q = st.session_state.original_question
            st.session_state.pending_clarification = None
            st.session_state.original_question = None
            
            with st.spinner("🧠 Alright, let me see what I can do..."):
                execute_query(original_q)
            
            st.session_state["_last_query_completed"] = time.time()
            st.rerun()

# =============================================
# INPUT BOX
# =============================================
if prompt := st.chat_input("Ex: What is the total revenue per region?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    clarification_result = check_needs_clarification(prompt)
    
    if clarification_result.needs_clarification:
        st.session_state.pending_clarification = clarification_result
        st.session_state.original_question = prompt

        st.session_state.messages.append({
            "role": "assistant",
            "content": clarification_result.message,
            "type": "clarification"
        })
        
        st.rerun()
    else:
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            with st.spinner("🧠 Thinking & Querying Database..."):
                execute_query(prompt)

            st.session_state["_last_query_completed"] = time.time()
            st.rerun()