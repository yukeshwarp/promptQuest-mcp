# Basic imports
import logging, re, uuid   
from functools import lru_cache   
from typing import List   
   
from cachetools import TTLCache, cached  #  NEW  
from mcp.server.fastmcp import FastMCP
from azure.mgmt.costmanagement.models import QueryDefinition
from azure.cosmos import CosmosClient, exceptions 
from cloud_config import CONTAINER_NAME, ENDPOINT, DATABASE_NAME, KEY, llmclient
import nltk
from nltk.corpus import stopwords
import re

# Download NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Initialize NLTK stopwords
stop_words = set(stopwords.words("english"))

# In-memory dataset cache  (max 32 datasets, 30-minute TTL each)  
DATASET_CACHE: TTLCache[str, List[str]] = TTLCache(maxsize=32, ttl=1800)   

def clean_text(text):
    """Remove non-alphanumeric characters and unnecessary spaces."""
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces
    text = text.strip()  # Remove leading/trailing whitespaces
    return text


def remove_stopwords(text):
    """Remove common stopwords from the text."""
    word_tokens = text.split(" ")
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return " ".join(filtered_text)


def preprocess_text(text):
    """Preprocess the text before sending it to the LLM."""
    text = clean_text(text)
    text = remove_stopwords(text)
    return text


# Initialize Cosmos DB client
client = CosmosClient(ENDPOINT, KEY)
database = client.get_database_client(DATABASE_NAME)
container = database.get_container_client(CONTAINER_NAME)

# Instantiate MCP server
mcp = FastMCP(
    name="CosmosDBQueryServer",
    host="0.0.0.0",  # Only used for SSE transport
    port=8050,  # Only used for SSE transport
)

# Helper that actually runs the Cosmos query   
def _run_cosmos_query(qdef: QueryDefinition | str):   
    return list(   
        container.query_items(qdef, enable_cross_partition_query=True)   
    )  
    
    
    
# 1) Dataset Loader Tools  

@mcp.tool()   
def load_dataset_by_date(start_date: str, end_date: str) -> str:   
    """   
    Pull all chat titles for the date window (inclusive) **once** and return a   
    dataset_id that can be reused later.   
    """   
    cache_key = f"date:{start_date}:{end_date}"   
    if cache_key in DATASET_CACHE:   
        logging.info("Using cached dataset %s", cache_key)   
        return cache_key   
    
    qdef = {
            "query": """
                SELECT c.ChatTitle, c.AssistantName
                FROM c
                WHERE c.TimeStamp BETWEEN @start AND @end
                ORDER BY c.TimeStamp DESC
            """,
            "parameters": [
                {"name": "@start", "value": start_date},
                {"name": "@end", "value": end_date}
            ]
        }
    try:   
        items = _run_cosmos_query(qdef)
        if not items:
            logging.warning("No data found for dates: %s to %s", start_date, end_date)
            DATASET_CACHE[cache_key] = []  # Cache empty list
            return cache_key   
        processed = [f"{preprocess_text(item["ChatTitle"])} || Agent:{item["AssistantName"]}" for item in items]
        DATASET_CACHE[cache_key] = processed   
        return cache_key   
    except exceptions.CosmosHttpResponseError as exc:   
        logging.exception("Cosmos query failed")   
        return f"ERROR:{exc.message}"   
   
@mcp.tool()   
def load_dataset_by_offset(offset: int, limit: int) -> str:
    """
    Fetch paginated chat entries by offset and limit
    Args:
        start_offset: Number of entries to skip
        limit: Maximum number of entries to return
    """
    offset = max(0, int(offset)); limit = min(int(limit), 20000)   
    cache_key = f"offset:{offset}:{limit}"   
    if cache_key in DATASET_CACHE:   
        logging.info("Using cached dataset %s", cache_key)   
        return cache_key   
   
    query = f"""   
        SELECT c.ChatTitle, c.AssistantName   
        FROM c   
        ORDER BY c.TimeStamp DESC   
        OFFSET {offset} LIMIT {limit}   
    """   
    try:   
        items = _run_cosmos_query(query)
        if not items:
            logging.warning("No data found for offset: %s and limit: %s", offset, limit)
            DATASET_CACHE[cache_key] = []  # Cache empty result
            return cache_key 
        processed = [f"{preprocess_text(item["ChatTitle"])} || Agent:{item["AssistantName"]}" for item in items]
        DATASET_CACHE[cache_key] = processed   
        return cache_key   
    except exceptions.CosmosHttpResponseError as exc:   
        logging.exception("Cosmos query failed")   
        return f"ERROR:{exc.message}"     
    



# 2) Analysis Tool – works only on cached data

@mcp.tool()   
def analyse_dataset(dataset_id: str, user_instruction: str) -> str:   
    """   
    Perform an LLM-powered analysis on an *already cached* dataset.   
    """   
    rows = DATASET_CACHE.get(dataset_id)
    if not rows:  # Check for empty list or None
        return "No data available for analysis."  
    if rows is None:   
        return f"Dataset '{dataset_id}' not found. Call load_dataset_* first."   
   
    # Clip to avoid token explosion   
    joined = ", ".join(rows)[:8000]   
   
    response = llmclient.chat.completions.create(   
        model="gpt-4.1",   
        messages=[   
            {"role": "system",   
             "content": "You are an expert data analysis."},   
            {"role": "user", "content": f"{user_instruction}\n\n{joined}"}   
        ],   
        temperature=0.3,   
    )   
    return response.choices[0].message.content.strip()  


# 3) Optional Flush Tool
  
@mcp.tool()   
def flush_dataset(dataset_id: str = "all") -> str:   
    """   
    Remove a dataset from the cache or flush everything.   
    """   
    if dataset_id == "all":   
        DATASET_CACHE.clear()   
        return "All datasets flushed."   
    if dataset_id in DATASET_CACHE:   
        del DATASET_CACHE[dataset_id]   
        return f"Dataset '{dataset_id}' flushed."   
    return f"Dataset '{dataset_id}' not present."  


# Run the MCP server
if __name__ == "__main__":
    # Clear all cached datasets before starting
    flush_dataset("all")  # Clear all cached datasets
    mcp.run(transport="sse")
