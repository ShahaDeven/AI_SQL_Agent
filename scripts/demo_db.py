import duckdb
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

db_path = os.path.join(PROJECT_ROOT, "data", "sql_agent_demo.db")

os.makedirs(os.path.dirname(db_path), exist_ok=True)

con = duckdb.connect(db_path)

print("Generating TPC-H Data (Scale Factor 0.1)...")
con.execute("INSTALL tpch; LOAD tpch;")
con.execute("CALL dbgen(sf=0.1)") 

print("Modifying Data (Adding Churn Risk)...")
con.execute("ALTER TABLE customer ADD COLUMN churn_risk VARCHAR")

con.execute("""
UPDATE customer
SET churn_risk = CASE 
    WHEN c_acctbal < 0 THEN 'HIGH_RISK' 
    WHEN c_acctbal < 3000 THEN 'MEDIUM_RISK'
    ELSE 'LOW_RISK'
END
""")

print("Modifying Data (Adding Promo Reduction)...")
con.execute("ALTER TABLE lineitem ADD COLUMN promo_reduction DOUBLE")
con.execute("UPDATE lineitem SET promo_reduction = 0.0") 
con.execute("UPDATE lineitem SET promo_reduction = 0.15 WHERE l_partkey IN (SELECT p_partkey FROM part WHERE p_type LIKE '%COPPER%')")

print("Calculating Total Value...")
con.execute("ALTER TABLE lineitem ADD COLUMN total_value DOUBLE")
con.execute("UPDATE lineitem SET total_value = l_extendedprice * (1 - l_discount)")

con.close()
print(f"Success! {db_path} created.")