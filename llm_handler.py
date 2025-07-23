from ctransformers import AutoModelForCausalLM
import threading

class LLMHandler:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LLMHandler, cls).__new__(cls)
                print("Initializing LLM Handler...")
                try:
                    # To use GPU, set gpu_layers > 0. For CPU, set to 0.
                    # This model will be downloaded automatically on the first run.
                    cls._instance.model = AutoModelForCausalLM.from_pretrained(
                        "TheBloke/sqlcoder-7B-GGUF",
                        model_type="llama",
                        gpu_layers=0 # Set to 0 for CPU, or a value > 0 for GPU layers
                    )
                    print("LLM model loaded successfully! 🧠")
                except Exception as e:
                    print(f"Error loading LLM model: {e}")
                    cls._instance.model = None
        return cls._instance

    def create_sql_query_prompt(self, question: str, db_schema: str) -> str:
        """
        Creates a more detailed and structured prompt for the LLM to generate an SQL query.
        """
        prompt = f"""You are a powerful text-to-SQL model. Your job is to generate a single, valid SQLite query to answer a user's question based on the provided database schema.

### Instructions:
- For questions about totals (like "total sales" or "total ad spend"), use the SUM() aggregate function.
- When using aggregate functions like SUM, COUNT, or AVG, always alias the column with a clear name using 'AS'. For example: 'SUM(total_sales) AS total_sales'.
- Ensure the generated query is a single, executable SQLite statement.
- Do not add any text before or after the SQL query. Do not add comments.

### Database Schema:
The query will run on a database with the following schema:
{db_schema}
### Question:
{question}

### SQL Query:
"""
        return prompt

    def generate_sql(self, question: str, db_schema: str) -> str:
        """
        Generates an SQL query from a natural language question using the improved prompt.
        """
        if not self.model:
            return "Error: Model not loaded"

        prompt = self.create_sql_query_prompt(question, db_schema)
        
        print("\n--- Generating SQL for question ---")
        print(f"Question: {question}")
        
        # The new prompt asks the model to generate the full query
        sql_query = self.model(prompt, max_new_tokens=256, temperature=0.0)
        
        # Clean up the generated query
        sql_query = sql_query.strip().replace("\n", " ").replace("`", "")
        if ";" in sql_query:
            sql_query = sql_query.split(";")[0]

        print(f"Generated SQL: {sql_query}")
        print("---------------------------------")
        return sql_query

# Instantiate the handler to be imported in other modules
llm_agent = LLMHandler()