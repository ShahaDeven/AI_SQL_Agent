import duckdb

db_path = 'D:\\Main_Python\\Projects\\AI_SQL_Agent\\data\\supply_chain.db'
con = duckdb.connect(db_path)
print(f"Connected to {db_path} for renaming...")

# ---------------------------------------------------------
# 1. AMBIGUITY INJECTION (The "Trap")
# We rename clear columns to vague ones.
# ---------------------------------------------------------
print("🔧 Renaming columns to create Semantic Ambiguity...")

# 'l_extendedprice' -> 'total_value' (Generic name)
# 'l_discount'      -> 'promo_reduction' (Non-standard name)
con.execute("ALTER TABLE lineitem RENAME COLUMN l_extendedprice TO total_value;")
con.execute("ALTER TABLE lineitem RENAME COLUMN l_discount TO promo_reduction;")

# Add a duplicate-sounding column to confuse the LLM
# Now we have 'total_value' AND 'total_value_forecast'
print("Creating difficult column (total_value_forecast)...")
con.execute("ALTER TABLE lineitem ADD COLUMN total_value_forecast DOUBLE;")
con.execute("UPDATE lineitem SET total_value_forecast = total_value * 1.1;") 

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING (The "Data Science" Part)
# We calculate 'churn_risk' based on account balance.
# ---------------------------------------------------------
print("Performing Feature Engineering: Calculating 'churn_risk'...")

# Add the column
con.execute("ALTER TABLE customer ADD COLUMN churn_risk VARCHAR;")

# Business Logic: Low Balance = High Risk
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