from agents.mcp import MCPServer, MCPUtil
from agents.models.chatcmpl_converter import Converter
from mcp.types import Tool as MCPTool
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam


async def get_mcp_tools(mcp_server: MCPServer) -> list[ChatCompletionToolParam]:
    server_tools = await mcp_server.list_tools()
    server_tools = [
        MCPUtil.to_function_tool(
            MCPTool(
                name=info.name,
                title=info.title,
                description=info.description,
                inputSchema=info.inputSchema,
                outputSchema=info.outputSchema,
                annotations=info.annotations,
            ),
            mcp_server,
            convert_schemas_to_strict=False,
        )
        for info in server_tools
    ]
    converted_tools = [Converter.tool_to_openai(tool) for tool in server_tools]

    return converted_tools
