# Basic imports
from mcp.server.fastmcp import FastMCP
from azure.cosmos import CosmosClient
from cloud_config import CONTAINER_NAME, ENDPOINT, DATABASE_NAME, KEY, llmclient
import nltk
from nltk.corpus import stopwords
import re

# Download NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Initialize NLTK stopwords
stop_words = set(stopwords.words("english"))


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

# Define tools for database queries

@mcp.tool()
def get_data_by_date(start_date_str: str, end_date_str: str) -> list:
    """
    Fetch chat titles between specified dates
    Args:
        start_date_str: Start date in format compatible with database
        end_date_str: End date in format compatible with database
    Returns:
        List of chat titles matching the date range
    """
    query = f"""
        SELECT c.ChatTitle 
        FROM c 
        WHERE c.TimeStamp BETWEEN '{start_date_str}' AND '{end_date_str}'
        ORDER BY c.TimeStamp DESC
    """
    items = []
    try:
        items = list(container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))
        # Preprocess the chat titles before returning
        items = [f"{preprocess_text(item["ChatTitle"])} || Agent:{item["AssistantName"]}" for item in items]
    except Exception as e:
        print(f"Error in get_data_by_date: {str(e)}")
    return items


@mcp.tool()
def get_data_by_entry_count(start_offset: int, limit: int) -> list:
    """
    Fetch paginated chat entries by offset and limit
    Args:
        start_offset: Number of entries to skip
        limit: Maximum number of entries to return
    Returns:
        List of chat entries with specified pagination
    """
    query = f"""
        SELECT c.id, c.TimeStamp, c.AssistantName, c.ChatTitle 
        FROM c 
        ORDER BY c.TimeStamp DESC 
        OFFSET {start_offset} LIMIT {limit}
    """
    items = []
    try:
        items = list(container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))
        # Preprocess the chat titles before returning
        items = [f"{preprocess_text(item["ChatTitle"])} || Agent:{item["AssistantName"]}" for item in items]
    except Exception as e:
        print(f"Error in get_data_by_entry_count: {str(e)}")
    return items


@mcp.tool()
def data_analysis_tool(processed_titles: list) -> str:
    """
    Analyze the provided chat titles using the Cosmos DB data.
    Args:
        processed_titles: List of chat titles to analyze (already preprocessed)
    Returns:
        Analysis result as a string
    """
    if not processed_titles:
        return "No titles to analyze."

    response = llmclient.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": "You're a very intelligent legal analytics assistant.",
            },
            {
                "role": "user",
                "content": f"""
                You are a legal domain expert extracting top 10 unique topics from user chat titles. Respond with the list only, no explanation.
                From the following user chat titles, identify and list the top 10 unique topics discussed. Do not add any explanation or extra words.

                Chat Titles:
                {processed_titles}
                """,
            },
        ],
        temperature=0.5,
        stream=False,
    )
    return response.choices[0].message.content.strip()


# Run the MCP server
if __name__ == "__main__":
    mcp.run(transport="sse")