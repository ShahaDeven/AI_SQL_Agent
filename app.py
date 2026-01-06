import streamlit as st
import duckdb
import pandas as pd
from src.agent_graph import agent_workflow,DB_PATH

def get_schema_for_ui():
    """
    Queries DuckDB to get a clean list of all tables and columns 
    so the user knows what they can ask about.
    """
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        
        # 1. Get all table names
        tables = con.execute("SHOW TABLES").fetchdf()
        
        schema_data = {}
        
        # 2. Loop through each table and get its columns
        for table_name in tables['name']:
            # DESCRIBE returns column_name, column_type, null, etc.
            columns = con.execute(f"DESCRIBE {table_name}").fetchdf()
            # We only keep the useful columns for the UI
            schema_data[table_name] = columns[['column_name', 'column_type']]
            
        con.close()
        return schema_data
    except Exception as e:
        return {"Error": str(e)}

# Page Config
st.set_page_config(page_title="AI SQL Agent", page_icon="🤖", layout="wide")

with st.sidebar:
    st.header("🗄️ Database Schema")
    st.markdown("Reference these tables when asking questions:")
    
    # 1. Define your descriptions (The "1-Liners")
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

    # 2. Load schema from Database
    schema_info = get_schema_for_ui()
    
    if "Error" in schema_info:
        st.error(f"Could not load schema: {schema_info['Error']}")
    else:
        # 3. Loop and Display
        for table_name, df_columns in schema_info.items():
            # Get the description or use a default if missing
            desc = table_descriptions.get(table_name, "No description available.")
            
            # Create a clean label: "CUSTOMER (👤 Registered users...)"
            with st.expander(f"**{table_name.upper()}**"):
                st.caption(desc) # Shows the summary in gray text
                st.dataframe(
                    df_columns, 
                    hide_index=True, 
                    use_container_width=True
                )

# Title and Header
st.title("🤖 AI SQL Agent")
st.markdown("Ask questions about your **Tpc-H Supply Chain Data** in plain English.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Check if this message is a DataFrame (Table)
        if message.get("type") == "dataframe":
            st.dataframe(message["content"], hide_index=True)
        else:
            # Standard text message
            st.markdown(message["content"])
            
        if "sql" in message:
            with st.expander("View Generated SQL"):
                st.code(message["sql"], language="sql")

# Input Box
if prompt := st.chat_input("Ex: How much revenue from high risk customers?"):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Answer
    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking & Querying Database..."):
            try:
                # Call the Agent
                response_data, sql_query = agent_workflow(prompt)
                
                # CASE 1: Response is a DataFrame (Success!)
                if isinstance(response_data, pd.DataFrame):
                    if not response_data.empty:
                        st.dataframe(
                            response_data, 
                            hide_index=True, 
                            use_container_width=True
                        )
                        # Store as dataframe
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
                    st.error(response_data) # Show the error in red
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_data,
                        "type": "text",
                        "sql": sql_query
                    })
                
                # CASE 3: Response is something else (None, etc.)
                else:
                    st.error("I could not generate a valid answer.")
                
                # Always show the SQL if it exists
                if sql_query:
                    with st.expander("View Generated SQL"):
                        st.code(sql_query, language="sql")
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")