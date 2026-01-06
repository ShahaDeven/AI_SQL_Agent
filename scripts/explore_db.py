import duckdb
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

db_path = os.path.join(PROJECT_ROOT, "data", "supply_chain.db")

con = duckdb.connect(db_path, read_only=True)

print(f"Connecting to {db_path}...\n")

tables = con.execute("SHOW TABLES").fetchall()
table_names = [t[0] for t in tables]
print(f"Found Tables: {table_names}\n")

target_tables = ['customer', 'lineitem']

for table in target_tables:
    print(f"--- INSPECTING: {table.upper()} ---")

    cols = con.execute(f"DESCRIBE {table}").fetchall()
    col_names = [c[0] for c in cols]
    
    print(f"Columns: {col_names}")
    
    # Show 3 rows to check data quality
    print("Sample Data:")
    rows = con.execute(f"SELECT * FROM {table} LIMIT 3").fetchall()
    for row in rows:
        print(row)
    print("\n")

# 3. Verify our Feature Engineering worked
print("--- VERIFYING CHURN RISK ---")
risk_dist = con.execute("SELECT churn_risk, COUNT(*) FROM customer GROUP BY churn_risk").fetchall()
print(f"Distribution: {risk_dist}")

con.close()