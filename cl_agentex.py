import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

SYSTEM_PROMPT = """You are a helpful AI assistant with access to specialized tools through MCP (Model Context Protocol).

CRITICAL: You MUST follow this exact format for ALL responses:

**MANDATORY FORMAT:**
1. First, always start with "Thought: " followed by your reasoning about the user's request
2. Then decide if you need tools or can answer directly
3. If using tools, specify the tool and parameters
4. AFTER getting tool results, provide a helpful summary/answer to the user

**For Tool Usage:**
Thought: [Your reasoning about what the user wants and why you need to use a tool]
Action: [Tool name]
Action Input: [Tool parameters]

**After Tool Execution:**
Based on the tool results, provide a clear, helpful response to the user's original question.

**For Direct Responses (no tools needed):**
Thought: [Your reasoning about why you can answer directly without tools]
Answer: [Your direct response]

**When to use tools:**
- Real-time data, current information
- File operations, database queries, web searches
- System interactions or external API calls
- Specialized computations or domain-specific operations

**When to respond directly:**
- General knowledge from your training
- Explanations, definitions, educational content
- Simple calculations or reasoning
- Advice or opinions based on your training

**IMPORTANT:** 
- Always start with "Thought:" to show your reasoning
- After using a tool, analyze the results and provide a meaningful response
- Don't just repeat the tool input/output - interpret and explain the results
- Be helpful and informative in your final response

Available tools and their purposes:
- SemanticSearch: Search through document collections using semantic similarity
- ForeignExchangeLookup: Get foreign exchange rates and currency information
- GetForeignExchangeTransactionData: Retrieve specific FX transaction data

Be thorough in your thinking process and always justify your approach."""

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
    model_client = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.1,
    )

    print("Loading tools from MCP servers...")
    tools = await multi_mcp_client.get_tools()
    print(f"Loaded {len(tools)} tools: {[tool.name for tool in tools]}")

    # Enhanced prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # Create the OpenAI tools agent
    agent = create_openai_tools_agent(model_client, tools, prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=10,
        early_stopping_method="generate",
        handle_parsing_errors=True,
    )

    # Interactive loop
    print("\n=== AI Assistant with MCP Tools (Thought/Action Format) ===")
    print("Available tools:", [tool.name for tool in tools])
    print("Type 'quit' to exit\n")
    
    while True:
        user_query = input("Enter your query: ")
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
            
        print(f"\n--- Processing: {user_query} ---")
        
        try:
            response = await agent_executor.ainvoke({
                "input": user_query,
                "chat_history": []
            })
            
            print("\n--- Agent Decision Process ---")
            
            # Parse the agent's initial reasoning (before tool execution)
            if "intermediate_steps" in response and response["intermediate_steps"]:
                # Show the reasoning process
                for idx, (action, observation) in enumerate(response["intermediate_steps"]):
                    print(f"\n🧠 Step {idx+1} - Agent Reasoning:")
                    
                    # Extract reasoning from the action log if available
                    if hasattr(action, 'log') and action.log:
                        reasoning_lines = action.log.split('\n')
                        for line in reasoning_lines:
                            line = line.strip()
                            if line.startswith("Thought:"):
                                print(f"   💭 {line}")
                            elif line.startswith("Action:"):
                                print(f"   🔧 {line}")
                            elif line.startswith("Action Input:"):
                                print(f"   📝 {line}")
                    
                    # Show tool execution
                    if hasattr(action, 'tool'):
                        print(f"   🔧 Executing Tool: {action.tool}")
                        print(f"   📝 Tool Input: {action.tool_input}")
                    
                    # Show tool results (truncated)
                    obs_str = str(observation)
                    if len(obs_str) > 500:
                        obs_str = obs_str[:500] + "..."
                    print(f"   📊 Tool Result: {obs_str}")
                    print("-" * 50)
            
            print("\n--- Final Response ---")
            final_output = response.get("output", "No output received")
            
            # Clean up the final output - remove repeated Thought/Action if present
            lines = final_output.split('\n')
            cleaned_lines = []
            skip_next = False
            
            for line in lines:
                line = line.strip()
                # Skip repeated Thought/Action/Action Input lines in final output
                if line.startswith(("Thought:", "Action:", "Action Input:")):
                    continue
                if line:
                    cleaned_lines.append(line)
            
            if cleaned_lines:
                print('\n'.join(cleaned_lines))
            else:
                print(final_output)
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("Please try again or check your MCP server connections.")
        
        print("\n" + "="*60)

    print("Goodbye!")

if __name__ == "__main__":
    asyncio.run(main())


# > Entering new AgentExecutor chain...
# Thought: The user wants to perform a semantic search on the term "april showers". I should use the SemanticSearch tool to perform this search.
# Action: SemanticSearch
# Action Input: {'message': 'april showers'}


# > Finished chain.

# --- Agent Decision Process ---

# --- Final Response ---
# Thought: The user wants to perform a semantic search on the term "april showers". I should use the SemanticSearch tool to perform this search.
# Action: SemanticSearch
# Action Input: {'message': 'april showers'}
