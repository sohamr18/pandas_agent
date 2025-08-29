from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import END, START
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
import os
import streamlit as st
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "TestProjectApp"

from langchain.chat_models import init_chat_model

llm = init_chat_model("groq:qwen/qwen3-32b")


def main():
    st.title("CSV Analyzer")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)
        # st.write("Preview of uploaded data:")
        # st.dataframe(df)

        llm = init_chat_model("groq:qwen/qwen3-32b")

        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            # agent_type=AgentType.OPENAI_FUNCTIONS,
            allow_dangerous_code=True,
        )

        user_question = st.text_input("Enter an exploratory data question")

        if st.button("Ask me"):
            with st.spinner("Processing..."):
                st.write(agent.invoke(user_question)["output"])
    else:
        st.info("Please upload a CSV file to get started.")
        # Create UI


if __name__ == "__main__":
    main()
