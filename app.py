import streamlit as st
import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from mcp_use import MCPAgent, MCPClient
from langchain_openai import AzureChatOpenAI

# ---------------------
# 🔧 Load environment
# ---------------------
load_dotenv()

# ---------------------
# 🔌 MCP Configuration
# ---------------------
MCP_SOURCE = "cosmosmcpserver"
DEFAULT_URL = "http://132.196.181.182:8050/sse"
MCP_ENV_VAR = "MCP_COSMOS_URL"

url = os.getenv(MCP_ENV_VAR, DEFAULT_URL)

# Include subscription & db/container info in config
config = {
    "mcpServers": {
        MCP_SOURCE: {
            "url": url,
            "type": "http"
        }
    }
}

# ---------------------
# 🤖 MCP Agent Function
# ---------------------
async def run_mcp_agent(user_query: str):
    client = MCPClient.from_dict(config)

    llm = AzureChatOpenAI(
        openai_api_key=os.getenv("LLM_KEY"),
        azure_endpoint=os.getenv("LLM_ENDPOINT"),
        deployment_name="gpt-4.1",
        api_version="2025-03-01-preview",
        model="gpt-4.1"
    )

    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=30,
    )

    final_prompt = f"""
            Answer the following query:

            \"\"\"{user_query}\"\"\"
            Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """

    return await agent.run(final_prompt)


def ask_mcp_agent(prompt):
    return asyncio.run(run_mcp_agent(prompt))

# ---------------------
# 📊 Streamlit UI
# ---------------------
st.set_page_config(page_title="ChatDB Analytics via MCP", layout="wide")
st.title("#ChatDB MCP Analytics")

# Session state
if "mcp_response" not in st.session_state:
    st.session_state["mcp_response"] = ""
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Sidebar input
with st.sidebar:
    st.markdown("### Ask About ChatDB")
    user_query = st.text_area("Enter your query (e.g., chat trend in Q1 2024)", height=100)

    if st.button("Run Analysis"):
        if user_query.strip():
            with st.spinner("Running LLM analysis via MCP..."):
                try:
                    result = ask_mcp_agent(user_query.strip())
                    st.session_state["mcp_response"] = result
                except Exception as e:
                    st.error(f"Agent error: {e}")
        else:
            st.warning("Please enter a query.")

# Main panel output
if st.session_state["mcp_response"]:
    st.subheader(" LLM-Based Analysis Result")
    st.markdown(st.session_state["mcp_response"])

# Interactive follow-up
if prompt := st.chat_input("Ask a follow-up or new question"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        try:
            followup_response = ask_mcp_agent(prompt)
        except Exception as e:
            followup_response = f"Agent error: {e}"

    st.session_state["messages"].append({"role": "assistant", "content": followup_response})
    with st.chat_message("assistant"):
        st.markdown(followup_response)

# Conversation history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])