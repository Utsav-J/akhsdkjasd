import json
import asyncio
import asyncio
from dotenv                         import load_dotenv
from langchain_mcp_adapters.client  import MultiServerMCPClient
from langgraph.prebuilt             import create_react_agent
from langchain_google_genai         import ChatGoogleGenerativeAI
from langchain_core.messages        import ToolMessage

class TestAgent:
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

    def __init__(self):
        load_dotenv()
        self.connection_info = {}
        self.message_history = []
        self.last_tool_call = None
        self.last_tool_context = None
        self.current_message_context_json = {}
        self.multi_mcp_config = {
            "mcp1": {
                "url": "http://localhost:8001/mcp",
                "transport": "streamable_http",
            },
            "mcp2": {
                "url": "http://localhost:8002/mcp",
                "transport": "streamable_http",
            },
        }
        self.multi_mcp_client = MultiServerMCPClient(self.multi_mcp_config)
        self.model_client = ChatGoogleGenerativeAI(model="gemini-2.0-flash", convert_system_message_to_human=True)
        self.connection_info = asyncio.run(self.create_mcp_session())

    def extract_tool_context(self, messages):
        tool_call_made = any(isinstance(item, ToolMessage) for item in messages)
        if not tool_call_made:
            return None, None
        tool_message = next((m for m in messages if isinstance(m, ToolMessage)), None)
        if not tool_message:
            return None, None
        try:
            tool_data = json.loads(tool_message.content)
            extracted_chunks = []
            document_urls = []
            for chunk in tool_data.get("result", {}).get("hits", []):
                title = chunk["record"].get("title", "Unknown Document")
                context = chunk["record"].get("raw_context", "")
                url = chunk["record"].get("url", "")
                if context:
                    extracted_chunks.append(f"📘 {title}\n{context}\n" + "*" * 20)
                if url and url not in document_urls:
                    document_urls.append(url)
            final_chunk_text = "\n\n".join(extracted_chunks) if extracted_chunks else None
            return final_chunk_text, document_urls
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            print(f"Error parsing tool message: {e}")
            return None, None

    def enhance_message_with_context(self, messages, extracted_context, document_urls):
        if not extracted_context:
            return messages
        enhanced_messages = messages.copy()
        context_message = {
            "role": "system",
            "content": f"""Based on the tool search results, here is the relevant context that should inform your response:\n\n        EXTRACTED CONTEXT:\n        {extracted_context}\n\n        DOCUMENT SOURCES:\n        {', '.join(document_urls) if document_urls else 'No URLs available'}\n\n        Please use this context to provide a comprehensive and accurate response to the user's query. Reference the specific information from these sources when relevant."""
        }
        enhanced_messages.append(context_message)
        return enhanced_messages

    def enhance_tool_context_json(self, messages):
        tool_message = next((m for m in messages if isinstance(m, ToolMessage)), None)
        if not tool_message:
            return None
        try:
            tool_data = json.loads(tool_message.content)
            result = tool_data.get("result")
            if not result:
                return None
            if (isinstance(result, dict) and "transactionId" in result) or (
                isinstance(result, list) and result and isinstance(result[0], dict) and "transactionId" in result[0]
            ):
                json_str = json.dumps(result, indent=2)
                system_message = {
                    "role": "system",
                    "content": (
                        "You have received the following Foreign Exchange Transaction Data from a tool call. "
                        "Represent this data as a table in your response. If there are nested fields, flatten them appropriately. "
                        "Here is the data (in JSON):\n\n"
                        f"{json_str}"
                    ),
                }
                self.current_message_context_json = result
                return system_message
            return None
        except Exception as e:
            print(f"Error in enhance_tool_context_json: {e}")
            return None

    async def create_mcp_session(self):
        try:
            tools = await self.multi_mcp_client.get_tools()
            agent = create_react_agent(self.model_client, tools, prompt=self.SYSTEM_PROMPT)
            return {
                'agent': agent,
                'multi_mcp_client': self.multi_mcp_client,
            }
        except Exception as e:
            print(f"Error creating Multi-MCP session: {e}")
            raise

    async def ask_agent(self, user_message: str) -> str:
        if not self.connection_info or 'agent' not in self.connection_info:
            raise RuntimeError("Agent not initialized. Please initialize the agent first.")
        try:
            if not self.message_history:
                self.message_history.append({"role": "system", "content": self.SYSTEM_PROMPT})
            self.message_history.append({"role": "user", "content": user_message})
            agent = self.connection_info['agent']
            response = await agent.ainvoke({"messages": self.message_history})
            full_messages = response.get("messages", []) if isinstance(response, dict) else []
            first_tool_message = next((m for m in full_messages if isinstance(m, ToolMessage)), None)
            new_tool_call_signature = None
            tool_context_to_store = None
            if first_tool_message:
                tool_name = getattr(first_tool_message, 'name', None)
                try:
                    tool_content = json.loads(first_tool_message.content)
                except Exception:
                    tool_content = first_tool_message.content
                if isinstance(tool_content, dict):
                    params = tuple(sorted((k, str(v)) for k, v in tool_content.items() if k != 'result'))
                else:
                    params = tuple()
                new_tool_call_signature = (tool_name, params)
                tool_context_to_store = tool_content
            context_reset = False
            is_followup = False
            if new_tool_call_signature and new_tool_call_signature != self.last_tool_call:
                context_reset = True
                self.last_tool_call = new_tool_call_signature
                self.last_tool_context = tool_context_to_store
                self.message_history = [{"role": "system", "content": self.SYSTEM_PROMPT}, {"role": "user", "content": user_message}]
                response = await agent.ainvoke({"messages": self.message_history})
                full_messages = response.get("messages", []) if isinstance(response, dict) else []
            elif new_tool_call_signature:
                self.last_tool_call = new_tool_call_signature
                self.last_tool_context = tool_context_to_store
            elif not new_tool_call_signature and self.last_tool_context:
                is_followup = True
            if is_followup:
                enhanced_messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
                json_str = json.dumps(self.last_tool_context, indent=2) if isinstance(self.last_tool_context, dict) else str(self.last_tool_context)
                enhanced_messages.append({
                    "role": "system",
                    "content": (
                        "You are answering a follow-up question. Here is the previous data context (in JSON):\n\n"
                        f"{json_str}\n\nUse this data to answer the user's question."
                    )
                })
                enhanced_messages.append({"role": "user", "content": user_message})
                final_response = await agent.ainvoke({"messages": enhanced_messages})
                response = final_response
                self.message_history.append({"role": "assistant", "content": str(response)})
                if isinstance(response, dict) and 'messages' in response:
                    return response['messages'][-1].content
                else:
                    return str(response)
            elif not full_messages:
                self.message_history.append({"role": "assistant", "content": str(response)})
                if isinstance(response, dict) and 'messages' in response:
                    return response['messages'][-1].content
                else:
                    return str(response)
            else:
                fx_tool_message = next((m for m in full_messages if isinstance(m, ToolMessage) and getattr(m, 'name', None) == 'GetForeignExchangeTransactionData'), None)
                if fx_tool_message:
                    system_message = self.enhance_tool_context_json(full_messages)
                    enhanced_messages = self.message_history.copy()
                    if system_message:
                        enhanced_messages.append(system_message)
                    final_response = await agent.ainvoke({"messages": enhanced_messages})
                    response = final_response
                    self.message_history.append({"role": "assistant", "content": str(response)})
                    if isinstance(response, dict) and 'messages' in response:
                        return response['messages'][-1].content
                    else:
                        return str(response)
                else:
                    extracted_context, document_urls = self.extract_tool_context(full_messages)
                    enhanced_messages = self.enhance_message_with_context(
                        self.message_history, extracted_context, document_urls
                    )
                    if extracted_context:
                        final_response = await agent.ainvoke({"messages": enhanced_messages})
                        response = final_response
                    self.message_history.append({"role": "assistant", "content": str(response)})
                    if isinstance(response, dict) and 'messages' in response:
                        return response['messages'][-1].content
                    else:
                        return str(response)
        except Exception as e:
            return f"❌ Sorry, I encountered an error: {str(e)}"

    async def main(self):
        print("🚀 Starting chat interface...")
        print("Type 'quit' to exit\n")
        while True:
            try:
                user_input = input("\n💬 You: ")
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                res = await self.ask_agent(user_input)
                print(f"\n{'='*50}")
                print("🤖 AGENT RESPONSE:")
                print(f"{'='*50}")
                print(res)
                print(f"\n{'='*50}")
                print(f"Current message context: {self.current_message_context_json}")
                print(f"{'='*50}\n")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    agent = TestAgent()
    res = asyncio.run(agent.ask_agent("Show me my approved transactions?"))
    print(res)