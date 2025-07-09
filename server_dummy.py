from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse

mcp = FastMCP(
    name="fisrt-server",
    host="127.0.0.1",
    port=8001,
)

DUMMY_POST_API_URL = "https://httpbin.org/post"


async def make_dummy_post_request(data: dict) -> dict:
    """Make a POST request to a dummy endpoint for testing."""

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(DUMMY_POST_API_URL, json=data, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Failed to contact dummy API: {str(e)}"}


@mcp.tool(name="SemanticSearch")
async def dummy_post_tool(message: str) -> Any:
    """
    Perform a semantic search on the vector database to retrieve data about april showers.
    When a user asks what are ___ questions, trigger this tool
    Args:
        message: Any string to send in the payload.
    """
    print("TOOL CALL")
    try:
        payload = {"message": message}
        dum_response = {
            "result": {
                "hits": [
                    {
                        "score": 0.7569691,
                        "record": {
                            "usecase_id": "GENAI101_CEOPT",
                            "document_id": "https://wellsfargo.bluematrix.com/links2/link/pdf/397f1b17-e968-4bfa-b245-2c4cdedabb0b",
                            "chunk_id": "120e06dcfad4882afc8b",
                            "raw_context": "Economics Special Commentary - March 25, 2025 April Showers For Better or Worse The first quarter of 2025 has been marked by several converging pressures that continue to shape our economic outlook. Persistent inflationary pressures in developed economies, particularly in the United States and European Union, have created a complex policy environment where central banks must balance growth concerns against price stability mandates. The Federal Reserve's recent decision to maintain interest rates at 5.25% has sent mixed signals to markets, with bond yields fluctuating between 4.2% and 4.6% throughout March. This volatility reflects deeper uncertainties about the sustainability of current monetary policy in an environment where core inflation remains stubbornly above the 2% target at 3.1%. Meanwhile, China's economic rebalancing continues to create ripple effects across global supply chains. The country's shift toward domestic consumption and away from export-driven growth has resulted in a 7% year-over-year decline in manufactured goods exports, particularly affecting electronics and automotive sectors worldwide.",
                            "file_name": "d853d45b-7b74-4608-9863-22369a6846b1.pdf",
                            "title": "https://wellsfargo.bluematrix.com/links2/link/pdf/397f1b17-e968-4bfa-b245-2c4cdedabb0b",
                            "data_classification": "internal",
                            "sor_last_modified": "2025-05-17T00:01:53.551391",
                            "book": "d853d45b-7b74-4608-9863-22369a6846b1",
                            "page_number": 1,
                            "file_id": "29a6ce0d-26c2-4cf3-86c8-f8ce14b2bc71",
                            "chunk_insert_date": "2025-05-15T04:32:44.644586",
                        },
                    },
                    {
                        "score": 0.7535724,
                        "record": {
                            "data_classification": "internal",
                            "sor_last_modified": "2025-05-17T00:01:53.551391",
                            "book": "d853d45b-7b74-4608-9863-22369a6846b1",
                            "page_number": 1,
                            "file_id": "29a6ce0d-26c2-4cf3-86c8-f8ce14b2bc71",
                            "chunk_insert_date": "2025-05-15T04:32:44.644586",
                            "usecase_id": "GENAI101_CEOPT",
                            "document_id": "https://wellsfargo.bluematrix.com/links2/link/pdf/241420e9247a49aadfa4",
                            "title": "https://wellsfargo.bluematrix.com/links2/link/pdf/397f1b17-e968-4bfa-b245-2c4cdedabb0b",
                            "chunk_id": "241420e9247a49aadfa4",
                            "raw_context": "April Showers Economics incredibly challenging to back into estimates of the economy. The `April Showers` economic environment of 2025 presents an unprecedented challenge for econometric modeling and forecasting, as traditional analytical frameworks struggle to capture the complex interplay of persistent inflation, geopolitical uncertainties, and rapid technological disruption that characterizes this transitional period. The volatile nature of current economic indicators—from fluctuating bond yields between 4.2% and 4.6% to unpredictable consumer spending patterns—creates a forecasting environment where historical correlations break down and standard regression models fail to provide reliable estimates. Much like predicting the exact timing and intensity of spring storms, economists find themselves grappling with non-linear relationships and structural breaks that make it incredibly difficult to back into coherent estimates of GDP growth, employment trends, or inflation trajectories, forcing analysts to rely more heavily on scenario planning and qualitative assessments rather than precise quantitative predictions during this period of economic turbulence.",
                            "file_name": "d853d45b-7b74-4608-9863-22369a6846b1.pdf",
                        },
                    },
                ]
            }
        }
        print(dum_response)
        return dum_response
    except:
        return {"Error": "But OK"}


# Add a custom GET /health route for health checks
@mcp.custom_route("/health", methods=["GET", "POST"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")


if __name__ == "__main__":
    transport = "sse"
    if transport == "stdio":
        print("Running dummy server with stdio transport")
        mcp.run(transport="stdio")
    elif transport == "sse":
        print("Running dummy server with SSE transport")
        mcp.run(transport="streamable-http")
    else:
        raise ValueError(f"Unknown transport: {transport}")
