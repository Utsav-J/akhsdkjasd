import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain import hub
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate

load_dotenv()

SYSTEM_PROMPT = """You are a helpful AI assistant with access to specialized tools through MCP (Model Context Protocol).

**When to use tools:**
- Use tools for tasks that require real-time data, external APIs, or specialized computations
- Use tools for file operations, database queries, web searches, or system interactions
- Use tools when you need to retrieve current information or perform actions you cannot do directly
- Use tools for domain-specific tasks that your available tools are designed for

**When to respond directly:**
- Answer general knowledge questions from your training data
- Provide explanations, definitions, or educational content
- Engage in conversation, creative writing, or brainstorming
- Perform simple calculations or reasoning that don't require external data
- Give advice or opinions based on your training

**Tool usage guidelines:**
- Always examine what tools are available to you first
- Use the most appropriate tool for the specific task
- Combine multiple tools if needed for complex workflows
- Explain your reasoning when choosing to use or not use tools
- If a tool fails, try alternative approaches or explain the limitation

Be efficient and thoughtful: use tools when they add value, but respond directly when you can provide accurate information from your knowledge base."""

async def main():
    # Configure both MCP servers
    multi_mcp_config = {
        "mcp1": {
            "url": "http://localhost:8001/mcp",
            "transport": "streamable_http",
        },
        "mcp2": {
            "url": "http://localhost:8002/mcp",
            "transport": "streamable_http",
        },
    }
    multi_mcp_client = MultiServerMCPClient(multi_mcp_config)
    model_client = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    print("Loading tools from MCP servers...")
    tools = await multi_mcp_client.get_tools()
    print(f"Loaded {len(tools)} tools.")

    # Use the OpenAI Functions agent for robust tool calling
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}\n\n{agent_scratchpad}")
    ])
    agent = create_openai_functions_agent(model_client, tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
    )

    user_query = input("Enter your query: ")
    print("\n--- Agent Reasoning Trace ---")
    response = await agent_executor.ainvoke({"input": user_query})
    if "intermediate_steps" in response:
        for idx, (action, observation) in enumerate(response["intermediate_steps"]):
            print(f"\nStep {idx+1}:")
            print(f"  Thought/Action: {getattr(action, 'log', str(action))}")
            print(f"  Observation: {observation}")
    print("\n--- Final Answer ---")
    print(response.get("output") or response)

if __name__ == "__main__":
    asyncio.run(main())
