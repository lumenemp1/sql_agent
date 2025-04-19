# agent.py

from llama_cpp import Llama
from langchain_core.language_models.llms import LLM
from typing import Optional, List, Any
from pydantic import PrivateAttr
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType

# ------------------ Config ------------------
LLAMA_MODEL_PATH = "sqlcoder-7b-2.Q4_K_M.gguf"
DB_URI = "mysql+pymysql://root:admin@localhost/chatbot"
N_CTX = 4096
N_THREADS = 8

# ------------------ Load SQLCoder GGUF ------------------
def load_llama(model_path: str):
    print(f"🧠 Loading SQLCoder model from {model_path} …")
    return Llama(
        model_path=model_path,
        n_gpu_layers=0,       # CPU-only
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        verbose=True,
        n_batch=512,
        use_mlock=True,
        use_mmap=True,
        logits_all=False,
    )

# ------------------ LangChain LLM Wrapper ------------------
class SQLCoderLLM(LLM):
    _model: Any = PrivateAttr()

    def __init__(self, model):
        super().__init__()
        self._model = model

    @property
    def _llm_type(self) -> str:
        return "sqlcoder-llama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        output = self._model(prompt, max_tokens=512, stop=stop)
        result = output["choices"][0]["text"]

        # 🛑 Clean multi-statement queries (only keep 1st statement)
        if "SELECT" in result and ";" in result:
            parts = result.split(";")
            result = parts[0].strip() + ";"

        return result.strip()

# ------------------ Main Agent Logic ------------------
def main():
    # Load SQLCoder model
    raw_model = load_llama(LLAMA_MODEL_PATH)
    llm = SQLCoderLLM(raw_model)

    # Connect to MySQL DB
    db = SQLDatabase.from_uri(DB_URI)

    # ✅ Patch schema method to skip row previews and fallback to column types
    def schema_only(self, table_names: Optional[List[str]] = None) -> str:
        from sqlalchemy import inspect

        inspector = inspect(self._engine)
        all_tables = inspector.get_table_names()

        if table_names is None:
            table_names = all_tables

        schema_str = ""
        for table in table_names:
            if table not in all_tables:
                continue

            try:
                # Attempt to get full DDL
                create_stmt = str(self._dialect.get_table_ddl(self._engine, table))
                schema_str += f"CREATE TABLE {table} ...\n{create_stmt}\n\n"
            except Exception:
                # Fallback to just column types
                try:
                    columns = inspector.get_columns(table)
                    col_defs = "\n  " + "\n  ".join(
                        f"{col['name']} {col['type']}" for col in columns
                    )
                    schema_str += f"CREATE TABLE {table} ...\n{col_defs}\n\n"
                except Exception:
                    schema_str += f"-- Unable to fetch schema for table {table}\n\n"

        return schema_str.strip()

    db.get_table_info = schema_only.__get__(db)

    # Create LangChain SQL Agent
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        agent_executor_kwargs={"handle_parsing_errors": True}
    )

    # User prompt
    question = input("❓ Ask a question about your database: ")

    # Get result
    result = agent_executor.invoke({"input": question})
    print("\n📊 Answer:")
    print(result)

# ------------------ Run ------------------
if __name__ == "__main__":
    main()
