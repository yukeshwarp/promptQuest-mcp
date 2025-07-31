import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

import os
import asyncio
from mcp_use import MCPAgent, MCPClient
from langchain_openai import AzureChatOpenAI

# Load environment variables
load_dotenv()

# MCP setup
MCP_SOURCE = "cosmosmcpserver"
DEFAULT_URL = "http://132.196.181.182:8050/sse"
MCP_ENV_VAR = "MCP_COSMOS_URL"
url = os.getenv(MCP_ENV_VAR, DEFAULT_URL)

config = {
    "mcpServers": {
        MCP_SOURCE: {
            "url": url,
            "type": "http"
        }
    }
}

# Async agent runner
async def run_mcp_agent(user_query: str):
    client = MCPClient.from_dict(config)
    llm = AzureChatOpenAI(
        openai_api_key=os.getenv("LLM_KEY"),
        azure_endpoint=os.getenv("LLM_ENDPOINT"),
        deployment_name="gpt-4.1",
        api_version="2025-03-01-preview",
        model="gpt-4.1"
    )
    agent = MCPAgent(llm=llm, client=client, max_steps=30)
    final_prompt = f"""
        Answer the following query:
        \"\"\"{user_query}\"\"\"
        Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    return await agent.run(final_prompt)

def ask_mcp_agent(prompt):
    return asyncio.run(run_mcp_agent(prompt))

# ----------------------------
# Initialize session state
# ----------------------------
if "chats" not in st.session_state: st.session_state["chats"] = []
if "messages" not in st.session_state: st.session_state["messages"] = []
if "Analysis" not in st.session_state: st.session_state["Analysis"] = ""
if "current_view" not in st.session_state: st.session_state["current_view"] = "Chat"

# ----------------------------
# Sidebar view switcher
# ----------------------------
with st.sidebar:
    st.radio(
        "Select View",
        options=["Chat", "Analytics"],
        index=0 if st.session_state["current_view"] == "Chat" else 1,
        key="current_view"
    )

# ----------------------------
# Main Title
# ----------------------------
st.title("Chat DB Analytics")

# ----------------------------
# === View: Chat ===
# ----------------------------
if st.session_state["current_view"] == "Chat":
    st.markdown('<div class="main-header">Interactive Chat Insights</div>', unsafe_allow_html=True)

    if "trend_analysis" in st.session_state and st.session_state["trend_analysis"]:
        st.write("### Trend Analysis")
        st.markdown(st.session_state["trend_analysis"])
        st.write("---")

    for message in st.session_state["messages"]:
        with st.chat_message(message.get("role", "user")):
            st.markdown(message.get("content", ""))

    if prompt := st.chat_input("Ask a question"):
        
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Optional topic summary if available
        topics_summary = ""
        topics = st.session_state.get("topics", [])
        if isinstance(topics, list) and topics:
            summary_labels = [t.get("label", "") for t in topics[:3] if t.get("label")]
            if summary_labels:
                topics_summary = "\n".join(f"- {label}" for label in summary_labels)

        mcp_query = f"""
        Use the following data from the chatbot usage database (focused on legal queries) to answer:

        Top topics summary:\n{topics_summary if topics_summary else '(No topics summary available)'}

        ---
        Prompt: {prompt}
        ---
        Provide an insightful answer using relevant data and context from chat titles.
        """
        with st.spinner("Thinking..."):
            try:
                response = ask_mcp_agent(mcp_query)
            except Exception as e:
                response = f"Agent error: {e}"

        st.session_state["messages"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
        

# ----------------------------
# === View: Analytics ===
# ----------------------------
elif st.session_state["current_view"] == "Analytics":
    st.subheader("Quarterly Topic Analysis")

    current_year = datetime.now().year
    selected_year = st.selectbox(
        "Select Year",
        options=[str(y) for y in range(current_year - 3, current_year + 1)],
        index=2,
    )

    quarters = {
        "Q1": (f"{selected_year}-01-01", f"{selected_year}-03-31"),
        "Q2": (f"{selected_year}-04-01", f"{selected_year}-06-30"),
        "Q3": (f"{selected_year}-07-01", f"{selected_year}-09-30"),
        "Q4": (f"{selected_year}-10-01", f"{selected_year}-12-31"),
    }

    def get_quarter_analysis(start_date, end_date):
        query = f"""
        Perform a legal chat usage analysis and extract top 10 unique legal topics discussed
        between {start_date} and {end_date}. Return only a clean list of topics.
        """
        try:
            return ask_mcp_agent(query)
        except Exception as e:
            return f"Agent error: {e}"

    q1, q2 = st.columns(2)
    q3, q4 = st.columns(2)

    with q1.container(height=500, border=True):
        st.markdown("**Q1 Topics**")
        with st.spinner("Loading Q1 data..."):
            st.write(get_quarter_analysis(*quarters["Q1"]))

    with q2.container(height=500, border=True):
        st.markdown("**Q2 Topics**")
        with st.spinner("Loading Q2 data..."):
            st.write(get_quarter_analysis(*quarters["Q2"]))

    with q3.container(height=500, border=True):
        st.markdown("**Q3 Topics**")
        with st.spinner("Loading Q3 data..."):
            st.write(get_quarter_analysis(*quarters["Q3"]))

    with q4.container(height=500, border=True):
        st.markdown("**Q4 Topics**")
        with st.spinner("Loading Q4 data..."):
            st.write(get_quarter_analysis(*quarters["Q4"]))
