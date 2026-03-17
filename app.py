import os
import warnings
import logging
import re
import streamlit as st
import streamlit.components.v1 as components
import duckdb
import pandas as pd
import time
import src.agent_graph as agent_graph_module
from src.clarifier import check_needs_clarification, refine_question

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

st.set_page_config(page_title="AI SQL Agent", page_icon="🤖", layout="wide")

USER_AVATAR = "🧑‍💼"
BOT_AVATAR = "🤖"

def scroll_to_bottom():
    """Inject JS to scroll the page to the bottom after rendering."""
    components.html(
        """
        <script>
            window.parent.document.querySelector('section.main').scrollTo(0, window.parent.document.querySelector('section.main').scrollHeight);
        </script>
        """,
        height=0,
    )

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

def is_simulation_result(df):
    """Check if a DataFrame looks like a simulation comparison result."""
    cols_lower = [c.lower() for c in df.columns]
    comparison_signals = ['original', 'simulated', 'difference', 'pct_change', 'percent_change',
                          'original_revenue', 'simulated_revenue', 'revenue_difference', 'scenario']
    matches = sum(1 for s in comparison_signals if any(s in c for c in cols_lower))
    return matches >= 2


def visualize_simulation(df):
    """Render simulation results with side-by-side comparison charts."""
    df.columns = df.columns.str.strip()
    cols_lower = {c.lower(): c for c in df.columns}
    
    # Detect if this is a sensitivity analysis (has scenario_label column)
    scenario_col = next((cols_lower[c] for c in cols_lower if 'scenario' in c), None)
    
    if scenario_col:
        # --- SENSITIVITY ANALYSIS VIEW ---
        st.caption("📊 Sensitivity Analysis — Multiple Scenarios")
        
        # Show the data table with styling
        st.dataframe(df, hide_index=True, width=1200)
        
        # Bar chart: scenario vs simulated values
        sim_col = next((cols_lower[c] for c in cols_lower if 'simulated' in c), None)
        orig_col = next((cols_lower[c] for c in cols_lower if 'original' in c), None)
        
        if sim_col and orig_col:
            chart_df = df[[scenario_col, orig_col, sim_col]].set_index(scenario_col)
            st.bar_chart(chart_df)
        
        # Show percent change as a line if available
        pct_col = next((cols_lower[c] for c in cols_lower if 'pct' in c or 'percent' in c), None)
        if pct_col:
            st.caption("📈 Impact Trend")
            st.line_chart(df.set_index(scenario_col)[[pct_col]])
        return
    
    # --- GROUPED COMPARISON VIEW (e.g., by region) ---
    text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    orig_col = next((cols_lower[c] for c in cols_lower if 'original' in c), None)
    sim_col = next((cols_lower[c] for c in cols_lower if 'simulated' in c), None)
    
    if text_cols and orig_col and sim_col:
        group_col = text_cols[0]
        
        st.caption(f"📊 Simulation Comparison — Original vs Simulated by {group_col}")
        
        # Side-by-side bar chart
        chart_df = df[[group_col, orig_col, sim_col]].set_index(group_col)
        st.bar_chart(chart_df)
        
        # Full data table below
        st.dataframe(df, hide_index=True, width=1200)
        return
    
    # --- SINGLE ROW COMPARISON (total level) ---
    if orig_col and sim_col:
        st.caption("📊 Simulation Result — Original vs Simulated")
        
        orig_val = df[orig_col].iloc[0]
        sim_val = df[sim_col].iloc[0]
        diff = sim_val - orig_val
        pct = (diff / orig_val * 100) if orig_val != 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Original", f"${orig_val:,.0f}")
        col2.metric("Simulated", f"${sim_val:,.0f}")
        col3.metric("Impact", f"{pct:+.2f}%", delta=f"${diff:,.0f}")
        
        st.dataframe(df, hide_index=True, width=1200)
        return
    
    # Fallback
    st.dataframe(df, hide_index=True, width=1200)


def auto_visualize(df, is_sim=False):
    """
    Analyzes the DataFrame and renders the best Streamlit chart.
    Routes simulation results to the specialized visualizer.
    """
    df.columns = df.columns.str.strip()

    # Check if this is a simulation result
    if is_sim or is_simulation_result(df):
        visualize_simulation(df)
        return

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

    st.dataframe(df, hide_index=True, width=1200)


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
                
                # Save simulation results to history for comparison
                if is_simulation_result(response_data):
                    # Create a short label from the prompt
                    label = prompt[:60] + "..." if len(prompt) > 60 else prompt
                    st.session_state["_scenario_history"].append({
                        "label": label,
                        "df": response_data.copy(),
                        "sql": sql_query,
                        "timestamp": time.strftime("%H:%M:%S"),
                    })
                    # Keep only last 10 scenarios
                    if len(st.session_state["_scenario_history"]) > 10:
                        st.session_state["_scenario_history"] = st.session_state["_scenario_history"][-10:]
                        
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
# DYNAMIC SCENARIO CONTEXT TRACKER
# =============================================
KNOWN_REGIONS = {"AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"}
KNOWN_SEGMENTS = {"AUTOMOBILE", "BUILDING", "FURNITURE", "HOUSEHOLD", "MACHINERY"}
KNOWN_NATIONS = {
    "ALGERIA", "ARGENTINA", "BRAZIL", "CANADA", "CHINA", "EGYPT", "ETHIOPIA",
    "FRANCE", "GERMANY", "INDIA", "INDONESIA", "IRAN", "IRAQ", "JAPAN", "JORDAN",
    "KENYA", "MOROCCO", "MOZAMBIQUE", "PERU", "ROMANIA", "RUSSIA",
    "SAUDI ARABIA", "UNITED KINGDOM", "UNITED STATES", "VIETNAM"
}

def extract_scenario_context(question: str):
    """
    Extract dimensions and percentages from a user query to adapt scenario presets.
    Zero API cost — pure regex/string matching.
    """
    q_upper = question.upper()
    ctx = {}

    # Extract region
    for r in KNOWN_REGIONS:
        if r in q_upper:
            ctx["region"] = r
            break

    # Extract nation
    for n in KNOWN_NATIONS:
        if n in q_upper:
            ctx["nation"] = n
            break

    # Extract segment
    for s in KNOWN_SEGMENTS:
        if s in q_upper:
            ctx["segment"] = s
            break

    # Extract price percentage
    price_match = re.search(r"price\s+(?:by\s+)?(\d+)%", question.lower())
    if price_match:
        ctx["price_pct"] = int(price_match.group(1))

    # Extract discount percentage
    disc_match = re.search(r"discount\s+(?:by\s+)?(\d+)%", question.lower())
    if disc_match:
        ctx["discount_pct"] = int(disc_match.group(1))

    # Generic percentage (fallback if no specific label)
    if "price_pct" not in ctx and "discount_pct" not in ctx:
        pct_match = re.search(r"(\d+)\s*%", question)
        if pct_match:
            pct_val = int(pct_match.group(1))
            if "price" in question.lower():
                ctx["price_pct"] = pct_val
            elif "discount" in question.lower():
                ctx["discount_pct"] = pct_val

    return ctx


def update_scenario_context(question: str):
    """Merge new context from the latest query into session state."""
    new_ctx = extract_scenario_context(question)
    if new_ctx:
        existing = st.session_state.get("_scenario_context", {})
        existing.update(new_ctx)
        st.session_state["_scenario_context"] = existing


# =============================================
# SIDEBAR
# =============================================
with st.sidebar:
    sidebar_tab = st.radio(
        "Navigation",
        ["📋 Schema", "🧪 Scenarios", "📊 Metrics"], 
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
                    st.dataframe(df_columns, hide_index=True, width=1200)

    elif sidebar_tab == "🧪 Scenarios":
        st.header("🧪 What-If Scenarios")
        st.markdown("Click a preset to run it instantly, or type your own.")

        # --- Dynamic Context from conversation ---
        ctx = st.session_state.get("_scenario_context", {})
        region = ctx.get("region", "EUROPE")
        nation = ctx.get("nation", None)
        segment = ctx.get("segment", None)
        price_pct = ctx.get("price_pct", 5)
        disc_pct = ctx.get("discount_pct", 10)

        # Display what context is active
        scope_label = region
        if nation:
            scope_label = nation
        if segment:
            scope_label = f"{segment} segment"

        if ctx:
            st.caption(f"📌 Adapted to your context: **{scope_label}** | Price: {price_pct}% | Discount: {disc_pct}%")
            if st.button("🔄 Reset to defaults", key="sc_reset"):
                st.session_state["_scenario_context"] = {}
                st.rerun()
            st.divider()

        # --- Scope target for prompts ---
        scope_filter = ""
        if nation:
            scope_filter = f" for the {nation} nation"
        elif region:
            scope_filter = f" for the {region} region"
        if segment:
            scope_filter = f" for the {segment} segment"

        st.subheader("💰 Pricing Scenarios")
        if st.button(f"📈 Price +{price_pct}%, show by region", key="sc_price_up"):
            st.session_state["_scenario_prompt"] = f"What if we increased the price by {price_pct}%{scope_filter}? Show the impact on revenue by region."
        if st.button(f"📉 Price -{price_pct}%, show by region", key="sc_price_down"):
            st.session_state["_scenario_prompt"] = f"What if we decreased the price by {price_pct}%{scope_filter}? Show the impact on revenue by region."

        st.subheader("🏷️ Discount Scenarios")
        if st.button(f"🏷️ Discount +{disc_pct}%, total impact", key="sc_disc_up"):
            st.session_state["_scenario_prompt"] = f"What if we increased the discount by {disc_pct}%? How would that affect total revenue?"
        if st.button(f"🏷️ Discount +{disc_pct}%{scope_filter} only", key="sc_disc_scoped"):
            st.session_state["_scenario_prompt"] = f"What if we increased the discount by {disc_pct}%{scope_filter} only? Show revenue by region comparing original vs simulated."

        st.subheader("🔀 Multi-Variable")
        if st.button(f"📈 Price +{price_pct}% AND Discount -{max(disc_pct // 3, 1)}%", key="sc_multi"):
            st.session_state["_scenario_prompt"] = f"What if we increased the price by {price_pct}% and reduced the discount by {max(disc_pct // 3, 1)}%{scope_filter}? Show revenue by region."

        st.subheader("📊 Sensitivity Analysis")
        if st.button("📊 Revenue sensitivity to discount", key="sc_sens_disc"):
            st.session_state["_scenario_prompt"] = f"How sensitive is total revenue to discount changes{scope_filter}? Test discount increases of 5%, 10%, 15%, and 20%."
        if st.button("📊 Revenue sensitivity to price", key="sc_sens_price"):
            st.session_state["_scenario_prompt"] = f"How sensitive is total revenue to price changes{scope_filter}? Test price increases of 5%, 10%, 15%, and 20%."

        # --- Scenario History & Comparison ---
        st.divider()
        st.subheader("📜 Scenario History")
        
        history = st.session_state.get("_scenario_history", [])
        
        if not history:
            st.caption("No scenarios run yet. Click a preset or type a What-If question.")
        else:
            st.caption(f"{len(history)} scenario(s) saved this session.")
            
            # Show each saved scenario
            for i, sc in enumerate(reversed(history)):
                with st.expander(f"🕐 {sc['timestamp']} — {sc['label']}"):
                    st.dataframe(sc["df"], hide_index=True, width=1200)
                    st.code(sc["sql"], language="sql")
            
            # --- Compare mode: select 2+ scenarios ---
            if len(history) >= 2:
                st.divider()
                st.subheader("⚖️ Compare Scenarios")
                
                # Let user pick which scenarios to compare
                scenario_labels = [f"{sc['timestamp']} — {sc['label']}" for sc in history]
                selected = st.multiselect(
                    "Select scenarios to compare:",
                    options=scenario_labels,
                    default=scenario_labels[-2:],  # Default to last 2
                    key="sc_compare_select"
                )
                
                if len(selected) >= 2:
                    # Find the selected scenarios
                    selected_scenarios = []
                    for label in selected:
                        for sc in history:
                            full_label = f"{sc['timestamp']} — {sc['label']}"
                            if full_label == label:
                                selected_scenarios.append(sc)
                                break
                    
                    # Try to build a comparison table
                    # Look for a common numeric column (simulated_value, revenue, etc.)
                    comparison_rows = []
                    for sc in selected_scenarios:
                        df = sc["df"]
                        cols_lower = {c.lower(): c for c in df.columns}
                        
                        # Find the key metric column
                        metric_col = None
                        for candidate in ['simulated_value', 'simulated_revenue', 'revenue', 'total_revenue']:
                            if candidate in cols_lower:
                                metric_col = cols_lower[candidate]
                                break
                        
                        if metric_col is None:
                            # Use first numeric column
                            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                            if numeric_cols:
                                metric_col = numeric_cols[0]
                        
                        if metric_col:
                            # Check if grouped (has text column)
                            text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
                            if text_cols:
                                # Grouped result — sum up for comparison
                                total = df[metric_col].sum()
                            else:
                                total = df[metric_col].iloc[0]
                            
                            comparison_rows.append({
                                "Scenario": sc["label"][:40],
                                "Value": round(total, 2),
                                "Time": sc["timestamp"]
                            })
                    
                    if comparison_rows:
                        comp_df = pd.DataFrame(comparison_rows)
                        
                        # Bar chart comparison
                        st.bar_chart(comp_df.set_index("Scenario")[["Value"]])
                        
                        # Data table
                        st.dataframe(comp_df, hide_index=True, width=1200)
                    else:
                        st.warning("Could not extract comparable metrics from selected scenarios.")
            
            # Clear history button
            if st.button("🗑️ Clear scenario history", key="sc_clear_history"):
                st.session_state["_scenario_history"] = []
                st.rerun()

    elif sidebar_tab == "📊 Metrics":
        st.header("📊 Agent Metrics")
        
        if st.button("🔄 Refresh Metrics"):
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

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = None

if "original_question" not in st.session_state:
    st.session_state.original_question = None

if "_scenario_context" not in st.session_state:
    st.session_state["_scenario_context"] = {}

if "_scenario_history" not in st.session_state:
    st.session_state["_scenario_history"] = []  # List of {label, df, sql, timestamp}

# Display Chat History
for idx, message in enumerate(st.session_state.messages):
    # Skip the last clarification message if we have a pending clarification
    # (it will be rendered by the pending clarification handler with buttons)
    if (st.session_state.pending_clarification and 
        message.get("type") == "clarification" and 
        idx == len(st.session_state.messages) - 1):
        continue
        
    icon = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=icon):
        if message.get("type") == "dataframe":
            auto_visualize(message["content"])
        elif message.get("type") == "clarification":
            # Display clarification message naturally
            st.markdown(message["content"])
        elif message.get("type") == "selection":
            # User's selection shown as a simple response
            st.markdown(f"**{message['content']}**")
        else:
            st.markdown(message["content"])

        if "sql" in message and message.get("sql"):
            with st.expander("View Generated SQL"):
                st.code(message["sql"], language="sql")

# Auto-scroll to the latest message
if st.session_state.messages:
    scroll_to_bottom()

# =============================================
# HANDLE PENDING CLARIFICATION
# =============================================
if st.session_state.pending_clarification:
    clarification = st.session_state.pending_clarification
    
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        # Conversational message
        st.markdown(clarification.message)
        
        st.markdown("") 
        
        # Create pill-style buttons in a row
        cols = st.columns(len(clarification.options))
        
        for i, option in enumerate(clarification.options):
            if cols[i].button(
                option,
                key=f"clarify_{i}"
            ):
                # User selected an option - refine the question
                original_q = st.session_state.original_question
                refined_q = refine_question(original_q, clarification.ambiguity_type, option)
                
                # Clear the pending clarification
                st.session_state.pending_clarification = None
                st.session_state.original_question = None
                
                # Add the selection to chat history (conversationally)
                st.session_state.messages.append({
                    "role": "user",
                    "content": option,
                    "type": "selection"
                })
                
                # Execute the refined query
                with st.spinner("🧠 Got it! Let me pull that data..."):
                    execute_query(refined_q)
                
                st.session_state["_last_query_completed"] = time.time()
                st.rerun()
        
        # Subtle skip option
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
# HANDLE SCENARIO PRESETS
# =============================================
if "_scenario_prompt" in st.session_state and st.session_state["_scenario_prompt"]:
    scenario_prompt = st.session_state.pop("_scenario_prompt")
    
    # Track context from preset
    update_scenario_context(scenario_prompt)
    
    # Add to chat as user message
    st.session_state.messages.append({"role": "user", "content": scenario_prompt})
    
    # Execute directly (no clarification needed for presets)
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.spinner("🧪 Running simulation..."):
            execute_query(scenario_prompt)
    
    st.session_state["_last_query_completed"] = time.time()
    st.rerun()

# =============================================
# INPUT BOX
# =============================================
if prompt := st.chat_input("Ex: What is the total revenue per region?"):
    # Track context from user query
    update_scenario_context(prompt)
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    # Check if clarification is needed (Rule-based, instant, free)
    clarification_result = check_needs_clarification(prompt)
    
    if clarification_result.needs_clarification:
        # Store the pending clarification state
        st.session_state.pending_clarification = clarification_result
        st.session_state.original_question = prompt
        
        # Add clarification request to chat history (natural message only)
        st.session_state.messages.append({
            "role": "assistant",
            "content": clarification_result.message,
            "type": "clarification"
        })
        
        st.rerun()
    else:
        # No clarification needed - execute directly
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            with st.spinner("🧠 Thinking & Querying Database..."):
                execute_query(prompt)

            st.session_state["_last_query_completed"] = time.time()
            st.rerun()