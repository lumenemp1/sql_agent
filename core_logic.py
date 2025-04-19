
import json
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama


# ----------------------------------------------------------------------
# Configuration – edit paths / model names here
# ----------------------------------------------------------------------
INDEX_PATH = "schema_index/faiss_index.bin"
META_PATH = "schema_index/table_metadata.json"
EMBED_MODEL_NAME = "BAAI/bge-small-en"
LLAMA_MODEL_PATH = "sqlcoder-7b-2.Q4_K_M.gguf"   # <- your GGUF file
TOP_K = 3                      # top‑K tables from semantic search
N_CTX = 2048                   # SQLCoder context window
N_THREADS = 6                  # adjust for your CPU
# ----------------------------------------------------------------------
from sqlalchemy import create_engine, text

# ✅ Your MySQL DB config — update as needed
DB_URI = "mysql+pymysql://root:admin@localhost/chatbot"


def load_faiss_and_metadata(index_path: str, meta_path: str):
    index = faiss.read_index(index_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return index, meta


def build_reverse_fk_map(metadata: dict) -> dict:
    """
    Returns a mapping: table_name -> set( tables_that_reference_it )
    """
    rev_map = {m["table_name"]: set() for m in metadata.values()}
    fk_pattern = re.compile(r"REFERENCES\s+`?(\w+)`?", re.IGNORECASE)

    for m in metadata.values():
        table = m["table_name"]
        ddl = m["create_stmt"]
        for ref in fk_pattern.findall(ddl):
            if ref in rev_map:
                rev_map[ref].add(table)
    return rev_map


def parse_forward_fks(ddl: str) -> set[str]:
    """Return set of table names referenced in this DDL (one hop)."""
    fk_pattern = re.compile(r"REFERENCES\s+`?(\w+)`?", re.IGNORECASE)
    return set(fk_pattern.findall(ddl))


def semantic_search(query: str, embed_model, faiss_index, top_k: int):
    q_emb = embed_model.encode(query)
    q_emb = np.array([q_emb], dtype="float32")
    _D, I = faiss_index.search(q_emb, top_k)
    return I[0]  # list of indices


def expand_with_related(idx_list, metadata, rev_fk_map):
    """
    Given initial FAISS indices, add tables that are FK‑related (one hop).
    """
    tables = {metadata[str(i)]["table_name"] for i in idx_list}
    extra = set()

    for i in idx_list:
        m = metadata[str(i)]
        table = m["table_name"]
        ddl = m["create_stmt"]

        # forward refs
        extra.update(parse_forward_fks(ddl))
        # reverse refs
        extra.update(rev_fk_map.get(table, set()))

    return tables.union(extra)


def build_schema_snippet(table_names: set[str], metadata: dict) -> str:
    """Concatenate CREATE TABLE statements for the chosen tables."""
    ddl_list = []
    # preserve original order for reproducibility
    for meta in metadata.values():
        if meta["table_name"] in table_names:
            ddl_list.append(meta["create_stmt"])
    return "\n\n".join(ddl_list)


def load_llama(model_path: str):
    print(f"Loading SQLCoder model from {model_path} …")
    return Llama(
        model_path=model_path,
        n_gpu_layers=0,       # CPU only
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        verbose=True,
        n_batch=512,
        use_mlock=True,
        use_mmap=True,
        logits_all=False,
    )


PROMPT_TEMPLATE = """### Task
Generate a SQL query to answer the following question:
{question}

### Database Schema
The query will run on a MySQL database with the following schema:
{schema}

### Guidelines
1. Ensure the query is fully compatible with MySQL.
2. Avoid using unsupported syntax, such as `NULLS LAST` or `NULLS FIRST`.
3. Handle `NULL` values explicitly using MySQL-compatible techniques (e.g., `CASE WHEN`).

### SQL Query
```sql
"""



# ----------------------------------------------------------------------
# Public Init and Inference Functions
# ----------------------------------------------------------------------

def init_models():
    global _faiss_index, _metadata, _rev_fk_map, _embed_model, _llm
    print("🔧 Initializing FAISS, Embeddings, SQLCoder...")
    _faiss_index, _metadata = load_faiss_and_metadata(INDEX_PATH, META_PATH)
    _rev_fk_map = build_reverse_fk_map(_metadata)
    _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    _llm = load_llama(LLAMA_MODEL_PATH)
    print("✅ Models loaded and ready!")





def process_question(question: str) -> dict:
    try:
        idxs = semantic_search(question, _embed_model, _faiss_index, TOP_K)
        final_tables = expand_with_related(idxs, _metadata, _rev_fk_map)
        schema_text = build_schema_snippet(final_tables, _metadata)
        prompt = PROMPT_TEMPLATE.format(question=question, schema=schema_text)

        final_sql = ""
        for chunk in _llm.create_completion(
            prompt,
            max_tokens=512,
            stop=["```"],
            temperature=0.1,
            stream=True,
        ):
            final_sql += chunk["choices"][0]["text"]

        final_sql = final_sql.strip()

        engine = create_engine(DB_URI)
        with engine.connect() as connection:
            result = connection.execute(text(final_sql))
            rows = [dict(row._mapping) for row in result.fetchall()]

        return {
            "sql": final_sql,
            "results": rows
        }

    except Exception as e:
        return {
            "sql": None,
            "results": [],
            "error": str(e)
        }
