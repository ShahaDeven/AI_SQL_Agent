import duckdb
import os

db_path = 'D:\\Main_Python\\Projects\\AI_SQL_Agent\\data\\supply_chain.db'

os.makedirs(os.path.dirname(db_path), exist_ok=True)

print(f"Initializing DuckDB at {db_path}...")
con = duckdb.connect(db_path)

print("Installing TPC-H extension...")
con.execute("INSTALL tpch; LOAD tpch;")

print("Generating TPC-H Data (Scale Factor=1). This takes ~30-60 seconds...")
con.execute("CALL dbgen(sf=1);") 

print("Data Generation Complete.")
con.close()