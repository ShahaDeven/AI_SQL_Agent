import duckdb
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

db_path = os.path.join(PROJECT_ROOT, "data", "supply_chain.db")

if not os.path.exists(db_path):
    raise FileNotFoundError(f"Database not found at {db_path}.")

con = duckdb.connect(db_path, read_only=True)

print(f"Connected to {db_path}...\n")

tables = con.execute("SHOW TABLES").fetchall()
table_names = [t[0] for t in tables]

print(f"Found {len(table_names)} Tables: {table_names}\n")


for table in table_names:
    print(f"--- 🔎 INSPECTING TABLE: {table.upper()} ---")
    
    schema = con.execute(f"DESCRIBE {table}").df()
    print("Schema:")
    print(schema[['column_name', 'column_type']].to_string(index=False))
    
    print("\nSample Data:")
    data = con.execute(f"SELECT * FROM {table} LIMIT 3").df()
    print(data.to_string(index=False))
    print("\n" + "="*50 + "\n")

con.close()