import duckdb
import os 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

db_path = os.path.join(PROJECT_ROOT, "data", "supply_chain.db")

con = duckdb.connect(db_path)
print(f"Connected to {db_path} for renaming...")

print("🔧 Renaming columns to create Semantic Ambiguity...")

con.execute("ALTER TABLE lineitem RENAME COLUMN l_extendedprice TO total_value;")
con.execute("ALTER TABLE lineitem RENAME COLUMN l_discount TO promo_reduction;")

print("Creating difficult column (total_value_forecast)...")
con.execute("ALTER TABLE lineitem ADD COLUMN total_value_forecast DOUBLE;")
con.execute("UPDATE lineitem SET total_value_forecast = total_value * 1.1;") 

print("Performing Feature Engineering: Calculating 'churn_risk'...")

con.execute("ALTER TABLE customer ADD COLUMN churn_risk VARCHAR;")

con.execute("""
    UPDATE customer 
    SET churn_risk = CASE 
        WHEN c_acctbal < 1000 THEN 'HIGH_RISK'
        WHEN c_acctbal < 5000 THEN 'MEDIUM_RISK'
        ELSE 'LOW_RISK'
    END;
""")

print("Modifying & Engineering Complete.")
con.close()