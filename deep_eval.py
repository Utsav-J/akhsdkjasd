import asyncio
from deepeval.tracing import observe, update_current_span
from deepeval.metrics import AnswerRelevancyMetric, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.dataset.golden import Golden
from deepeval.models import GeminiModel
from deepeval import evaluate
from cl_agent import TestAgent

model = GeminiModel(
    model_name="gemini-2.0-flash", api_key="AIzaSyCijt8NxzuMa4ROq-Z_basaWHtECy05bqs"
)
# 1. Initialize your agent
agent = TestAgent()

# 2. Define a wrapper for DeepEval
@observe(
    type="agent",
    metrics=[AnswerRelevancyMetric(model=model), ToolCorrectnessMetric()]
)
async def agent_wrapper( user_input: str, expected_tools=None, expected_output=None):
    # Run the agent and get the output
    output = await agent.ask_agent(user_input)
    # Optionally, you can parse tool calls from the agent if you expose them
    # For demonstration, let's assume you can get tool calls from agent.message_history
    tools_called = []
    for msg in agent.message_history:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            tools_called.append(ToolCall(name=msg.get("name", "UnknownTool")))
    # Update DeepEval's span with the test case
    update_current_span(
        test_case=LLMTestCase(
            input=user_input,
            actual_output=output,
            tools_called=tools_called,
            expected_tools=expected_tools or [],
            expected_output=expected_output
        )
    )
    return output

# 3. Define your test cases (goldens)
goldens = [
    Golden(
        input="Show me my approved transactions?",
        expected_output="Here are your approved transactions",  # Adjust as needed
        expected_tools=[ToolCall(name="GetApprovedTransactions")]
    ),
    # Add more test cases as needed
]

# 4. Run the evaluation
evaluate(goldens=goldens, observed_callback=agent_wrapper)
