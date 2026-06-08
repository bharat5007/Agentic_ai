import truststore


from fastmcp import FastMCP

truststore.inject_into_ssl()
mcp = FastMCP("Simple Calculator Server")
