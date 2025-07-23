import os
import asyncio
import matplotlib.pyplot as plt
import pandas as pd
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse
from sqlalchemy import create_engine, text, inspect

# Local imports from other project files
from llm_handler import llm_agent

# --- App and Database Configuration ---
app = FastAPI(
    title="E-commerce AI Agent",
    description="An AI agent to answer questions about e-commerce data via API.",
)

DB_NAME = "ecommerce.db"
engine = create_engine(f"sqlite:///{DB_NAME}")
CHART_DIR = "charts"

# Create the directory for charts if it doesn't already exist
os.makedirs(CHART_DIR, exist_ok=True)

# --- Helper Functions ---

def get_db_schema():
    """
    Inspects the database and returns a CREATE TABLE string for each table.
    """
    inspector = inspect(engine)
    schema = ""
    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        schema += f"CREATE TABLE {table_name} (\n"
        for column in columns:
            schema += f"  {column['name']} {column['type']},\n"
        schema = schema.rstrip(",\n") + "\n);\n\n"
    return schema

def execute_sql_query(query: str):
    """
    Executes the provided SQL query against the database and returns the result.
    """
    try:
        with engine.connect() as connection:
            if any(keyword in query.upper() for keyword in ["UPDATE", "INSERT", "DELETE"]):
                 result = connection.execute(text(query))
                 connection.commit()
                 return pd.DataFrame([{"status": "Success", "rows_affected": result.rowcount}])
            else:
                df = pd.read_sql_query(text(query), connection)
                return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid SQL Query or database error: {e}")

def format_response_as_string(question: str, df: pd.DataFrame) -> str:
    """
    Formats the DataFrame result into a human-readable string.
    """
    if df.empty:
        return f"I couldn't find any data for your question: '{question}'."
    
    q_lower = question.lower()

    if "total sales" in q_lower:
        total = df.iloc[0,0]
        return f"Your total sales are ${total:,.2f}."

    if "roas" in q_lower or "return on ad spend" in q_lower:
        roas = df.iloc[0,0]
        return f"Your Return on Ad Spend (RoAS) is {roas:.2f}."

    if "cpc" in q_lower or "cost per click" in q_lower:
        item_id = df.iloc[0, 0]
        cpc = df.iloc[0, 1]
        return f"The item with the highest Cost Per Click (CPC) is item_id '{item_id}' with a CPC of ${cpc:.2f}."

    return f"Here is the data for your question '{question}':\n{df.to_string()}"

def generate_chart(df: pd.DataFrame, question: str) -> str | None:
    """
    Generates a chart if the query result is suitable for visualization.
    """
    if len(df) > 1 and len(df.columns) == 2:
        try:
            x_col, y_col = df.columns[0], df.columns[1]
            
            # Correction: Ensure the x-axis column is treated as a string for plotting labels,
            # which works for both text and numerical IDs.
            df[x_col] = df[x_col].astype(str)

            # Check if the y-axis is numeric.
            if pd.api.types.is_numeric_dtype(df[y_col]):
                plt.figure(figsize=(12, 7))
                plt.bar(df[x_col], df[y_col], color='#5DADE2')
                plt.xlabel(x_col.replace('_', ' ').title(), fontsize=12)
                plt.ylabel(y_col.replace('_', ' ').title(), fontsize=12)
                plt.title(question.title(), fontsize=16)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                chart_path = os.path.join(CHART_DIR, "latest_chart.png")
                plt.savefig(chart_path)
                plt.close()
                
                print(f"Chart generated and saved to {chart_path}")
                return chart_path
        except Exception as e:
            print(f"Could not generate chart: {e}")
    return None

async def stream_answer(response: dict):
    """
    Generator function for streaming the response to simulate a live typing effect.
    """
    answer_words = response["answer"].split()
    for word in answer_words:
        yield f"{word} "
        await asyncio.sleep(0.05)

# --- API Endpoints ---

@app.get("/", summary="Root Endpoint", description="Provides a welcome message for the API.")
def read_root():
    return {"message": "Welcome to the E-commerce AI Agent API!"}

@app.post("/ask", summary="Ask a question about the data", description="Receives a question, generates SQL, queries the database, and returns the answer.")
async def ask_question(request: dict = Body(...)):
    """
    This endpoint processes a user's question, generates SQL, queries the database,
    and returns a response including a human-readable answer and a chart if applicable.
    """
    question = request.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="'question' field is required.")

    db_schema = get_db_schema()
    sql_query = llm_agent.generate_sql(question, db_schema)
    
    if "error" in sql_query.lower():
         raise HTTPException(status_code=500, detail="Failed to generate a valid SQL query.")

    # Step 1: Execute the query to get the DataFrame
    result_df = execute_sql_query(sql_query)
    
    # Step 2: Pass the DataFrame to the chart generator
    chart_path = generate_chart(result_df, question)
    
    # Step 3: Pass the DataFrame to the response formatter
    human_readable_answer = format_response_as_string(question, result_df)

    # Step 4: Return everything
    return {
        "question": question,
        "answer": human_readable_answer,
        "generated_sql": sql_query,
        "chart_url": chart_path
    }

@app.post("/ask-stream", summary="Ask a question with a streaming response", description="Provides a live typing effect for the answer.")
async def ask_question_stream(request: dict = Body(...)):
    """
    This bonus endpoint provides the answer using event streaming.
    """
    response_data = await ask_question(request)
    return StreamingResponse(stream_answer(response_data), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    if not os.path.exists(DB_NAME):
        print("Database not found! Please run 'python database_setup.py' first.")
    else:
        print("Starting FastAPI server... Visit http://127.0.0.1:8000/docs for the API documentation.")
        uvicorn.run(app, host="0.0.0.0", port=8000)