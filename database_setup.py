import pandas as pd
from sqlalchemy import create_engine, text
import os

# --- Configuration ---
DB_NAME = "ecommerce.db"
DATA_DIR = "data"
# Match the filenames you uploaded
CSV_FILES = {
    "eligibility": os.path.join(DATA_DIR, "Product-Level Eligibility Table (mapped) - Product-Level Eligibility Table (mapped).csv"),
    "ad_metrics": os.path.join(DATA_DIR, "Product-Level Ad Sales and Metrics (mapped) - Product-Level Ad Sales and Metrics (mapped).csv"),
    "sales_metrics": os.path.join(DATA_DIR, "Product-Level Total Sales and Metrics (mapped) - Product-Level Total Sales and Metrics (mapped).csv"),
}

# --- Database Engine ---
engine = create_engine(f"sqlite:///{DB_NAME}")

def setup_database():
    """
    Reads the provided CSV files and loads their data into SQLite tables.
    Deletes the database file if it already exists to ensure a fresh start.
    """
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
        print(f"Removed existing database: {DB_NAME}")

    print("Creating new database and tables from your files...")
    try:
        with engine.connect() as connection:
            for table_name, file_path in CSV_FILES.items():
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    # Clean up column names for easier SQL queries (remove spaces, etc.)
                    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
                    df.to_sql(table_name, con=connection, if_exists="replace", index=False)
                    print(f"  - Table '{table_name}' created and populated successfully.")
                else:
                    print(f"  - Error: File not found at {file_path}. Please check the path and filename.")
            
            # Print table schemas for verification
            print("\n--- Database Schema ---")
            inspector = text("SELECT name FROM sqlite_master WHERE type='table';")
            tables = connection.execute(inspector).fetchall()
            for table in tables:
                table_name = table[0]
                result = connection.execute(text(f"PRAGMA table_info({table_name});"))
                columns = result.fetchall()
                print(f"\nTable: {table_name}")
                for col in columns:
                    print(f"  - {col[1]} ({col[2]})")

        print("\nDatabase setup complete! ✨")

    except Exception as e:
        print(f"An error occurred during database setup: {e}")

if __name__ == "__main__":
    setup_database()