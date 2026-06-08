import random
import json
from fastmcp import FastMCP

mcp = FastMCP(name="Demo Server")


@mcp.tool
def roll_dice(n_dice: int = 1) -> list[int]:
    """Roll n_dice 6-sided dice and return the result"""
    return [random.randint(1, 6) for _ in range(n_dice)]


@mcp.tool
def add_numers(a: float, b: float) -> float:
    """Add two numbers together"""
    return a + b


@mcp.resource("info://server")
def server_info() -> str:
    """Get information about this server"""
    info = {
        "name": "Simple calculator server",
        "version": "1.0.0",
        "description": "A basic mcp server with math tools",
        "tools": ["add", "roll_dice"],
        "author": "Bharat",
    }
    return json.dumps(info, indent=2)


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
