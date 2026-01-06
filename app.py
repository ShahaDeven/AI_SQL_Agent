import os
import warnings
import logging
import streamlit as st
import duckdb
import pandas as pd
from src.agent_graph import agent_workflow, DB_PATH
import src.agent_graph as agent_graph_module

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger('tensorflow').setLevel(logging.ERROR)

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', module='tensorflow')

os.environ["CHROMA_TELEMETRY_IMPL"] = "chromadb.telemetry.posthog.Posthog" 
os.environ["ANONYMIZED_TELEMETRY"] = "False"

st.set_page_config(page_title="AI SQL Agent", page_icon="🤖", layout="wide")


@st.cache_resource
def load_agent_resources():
    print("⏳ Loading Heavy AI Models... (This should only happen once)")
    return agent_graph_module

agent_module = load_agent_resources()
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


with st.sidebar:
    st.header("🗄️ Database Schema")
    st.markdown("Reference these tables when asking questions:")
    
    table_descriptions = {
        "customer": "👤 Registered users, account balances, and churn risk.",
        "orders": "📦 Order headers, dates, and priority status.",
        "lineitem": "🧾 Individual items in an order (price, discount, promo).",
        "nation": "🌍 Countries associated with customers and suppliers.",
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


st.title("🤖 AI SQL Agent")
st.markdown("Ask questions about your **Supply Chain Data** in plain English.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("type") == "dataframe":
            auto_visualize(message["content"])
        else:
            st.markdown(message["content"])

        if "sql" in message:
            with st.expander("View Generated SQL"):
                st.code(message["sql"], language="sql")

# Input Box
if prompt := st.chat_input("Ex: What is the total revenue per region?"):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Answer
    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking & Querying Database..."):
            try:
                # Call the Agent
                response_data, sql_query = agent_module.agent_workflow(prompt)
                
                # CASE 1: Response is a DataFrame (Success!)
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
                
                # CASE 2: Response is a String (Error Message or Text Answer)
                elif isinstance(response_data, str):
                    st.error(response_data) 
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_data,
                        "type": "text",
                        "sql": sql_query
                    })
                
                # CASE 3: Response is something else
                else:
                    st.error("I could not generate a valid answer.")

                if sql_query:
                    with st.expander("View Generated SQL"):
                        st.code(sql_query, language="sql")
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")