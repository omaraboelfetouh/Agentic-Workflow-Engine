import inspect
from typing import Callable, Dict, Any

class ToolRegistry:
    "\""
    A registry for managing and executing tools available to autonomous AI agents.
    "\""
    def __init__(self):
        self._tools: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable):
        "\""Registers a function as a tool."\""
        self._tools[name] = func
        print(f"Tool '{name}' registered successfully.")

    def execute(self, name: str, **kwargs) -> Any:
        "\""Executes a registered tool by name with provided arguments."\""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found in registry.")
        
        func = self._tools[name]
        sig = inspect.signature(func)
        
        # Filter kwargs to only those accepted by the function
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        
        try:
            return func(**filtered_kwargs)
        except Exception as e:
            return f"Error executing tool '{name}': {str(e)}"

    def list_tools(self) -> list:
        return list(self._tools.keys())

# Example Usage
if __name__ == "__main__":
    registry = ToolRegistry()
    
    def search_web(query: str) -> str:
        return f"Mock search results for: {query}"
        
    registry.register("search", search_web)
    print(registry.execute("search", query="Latest AI advancements"))