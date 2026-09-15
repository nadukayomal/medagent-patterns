"""
Web Search Tool Module.

This module implements the `WebSearchTool` class, which serves as a wrapper around the Tavily API
to execute web searches. It is designed to be utilized by AI agents within the Agentic RAG system
when they require real-time or external information that is not available in the internal knowledge base.

Overall, this file provides the structured interface necessary to plug external web search capabilities 
directly into the LLM's 'Tool Use' and 'ReAct' design patterns, formatting search results into 
context-rich strings that language models can easily parse and reason over.
"""

import time
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from zoneinfo import ZoneInfo
from tavily import TavilyClient

# Add the project root to sys.path before trying to import custom packages
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import get_api_key, get_model_configuration, get_parameter_configuration, ToolRegistry


class WebSearchTool:
    """
    A tool class for performing web searches using the Tavily API.

    This class handles the initialization of the API client and provides methods to perform searches,
    process the raw API responses, format them for LLM context, and expose the tool's JSON schema 
    definition so it can be registered with language models.
    """

    def __init__(
                    self, 
                    max_results: Optional[int] = None, 
                    prefer_domains: Optional[List[str]] = None, 
                    freshness_days: Optional[int] = None, 
                    search_depth: str = "advanced"
                    ):
        """
        Initializes the WebSearchTool with the specified configurations and sets up the Tavily client.

        Args:
            max_results (Optional[int]): The maximum number of search results to return.
            prefer_domains (Optional[List[str]]): A list of specific domains to prioritize in the search results.
            freshness_days (Optional[int]): The number of days back to search for recent content.
            search_depth (str): The depth of the search to perform (e.g., "basic" or "advanced"). Defaults to "advanced".
        """
        params = get_parameter_configuration()
        self.max_results = max_results or params.get("web.max_results", 5)
        self.prefer_domains = prefer_domains or params.get("web.prefer_domains", [".lk", ".gov", ".org"])
        self.freshness_days = freshness_days or params.get("web.freshness_days", 120)
        self.search_depth = search_depth or params.get("web.search_depth", "advanced")

        self.api_key = get_api_key("tavily")
        self.client = TavilyClient(api_key = self.api_key)
        self.timezone = ZoneInfo(params.get("output", {}).get("timezone", "Asia/Colombo"))

    def search(
                self, 
                query: str,
                max_results: Optional[int] = None, 
                include_domains: Optional[List[str]] = None
                ) -> Dict[str, Any]:
        """
        Executes a web search query using the Tavily API.

        Args:
            query (str): The specific search query string to look up.
            max_results (Optional[int]): Overrides the default maximum number of results for this specific search.
            include_domains (Optional[List[str]]): A list of domains to strictly include/limit to for this search.

        Returns:
            Dict[str, Any]: The raw search response dictionary returned directly by the Tavily API.
        """
        start_time = time.time()

        # Prepare search parameters
        search_params = {
                            "query": query,
                            "max_results": max_results or self.max_results,
                            "search_depth": self.search_depth,
                            "include_answer": True,
                            "include_raw_content": False
                            }

        if include_domains:
            search_params["include_domains"] = include_domains

        try:
            response = self.client.search(**search_params)
            results = self._process_results(response)
            end_time = time.time()

            latency_ms = int((end_time - start_time) * 1000)

            return {
                "status": "success",
                "query": query,
                "results": results,
                "answer": response.get("answer", ""),
                "result_count": len(results),
                "latency_ms": latency_ms,
                "checked_at": datetime.now(self.timezone).strftime("%Y-%m-%d %H:%M:%S %Z")
            }
        
        except Exception as e:
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)

            return {
                "status": "error",
                "query": query,
                "error": str(e),
                "results": [],
                "latency_ms": latency_ms
            }


    def _process_results(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Processes and sanitizes the raw response received from the Tavily API.

        This is an internal helper method used to clean up the raw data, extract the most relevant
        snippets, and discard unnecessary metadata before formatting.

        Args:
            response (Dict[str, Any]): The raw JSON/dictionary response from the Tavily API.

        Returns:
            List[Dict[str, Any]]: A processed list of dictionaries where each dictionary represents 
                                  a clean, relevant search result (e.g., URL, title, content).
        """
        results = response.get("results", [])

        if not results:
            return []
        
        # Score results based on preferences
        scored_results = []

        for result in results:
            url = result.get("url", "")
            score = result.get("score", 0.5)
            
            # Boost preferred domains
            for domain in self.prefer_domains:
                if domain in url:
                    score += 0.2
                    break
            
            # Check if government or official site
            if ".gov" in url or "official" in url.lower():
                score += 0.15
            
            scored_results.append({
                "title": result.get("title", ""),
                "url": url,
                "content": result.get("content", ""),
                "score": min(score, 1.0),  # Cap at 1.0
                "published_date": result.get("published_date", "")
            })

         # Sort by score (descending)
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_results
        
    def format_result(
                        self, 
                        search_response: Dict[str, Any], 
                        include_urls: bool = True) -> str:
        """
        Formats the processed search results into a readable string format optimized for LLMs.

        Args:
            search_response (Dict[str, Any]): The raw or processed search response data to be formatted.
            include_urls (bool): Whether to include the source URLs in the formatted string. Defaults to True.

        Returns:
            str: A formatted string containing the concatenated titles, snippets, and optionally URLs,
                 ready to be injected into an LLM's prompt context.
        """
        if search_response["status"] == "error":
            return f"Web search failed: {search_response.get('error', 'Unknown error')}"

        results = search_response["results"]

        if not results:
            return (
                "No reliable web results found. "
                "Please verify information by calling the official hospital hotline."
            )

        output_parts = []

        # Add Tavily's answer summary if available
        if search_response.get("answer"):
            output_parts.append(f"Summary: {search_response['answer']}\n")

        # Add individual results
        output_parts.append("External Web Sources:\n")

        for idx, result in enumerate(results[:self.max_results], 1):
            title = result["title"]
            content = result["content"][:300] + "..." if len(result["content"]) > 300 else result["content"]
            
            result_text = f"{idx}. {title}\n{content}"
            
            if include_urls:
                result_text += f"\nURL: {result['url']}"
            
            output_parts.append(result_text)

        # Add metadata
        checked_at = search_response.get("checked_at", "")
        if checked_at:
            output_parts.append(f"\nInformation checked at: {checked_at}")
        
        return "\n\n".join(output_parts)
        
    def get_tool_definition(self) -> Dict[str, Any]:
        """
        Generates the standard JSON schema definition for this tool.

        This method is critical for the 'Tool Use' design pattern, as it tells the LLM exactly 
        what this tool does, what arguments it accepts, and what types those arguments should be.

        Returns:
            Dict[str, Any]: A dictionary representing the JSON schema of the search tool, structured 
                            according to OpenAI's tool calling specifications.
        """
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the web for current, real-world information about healthcare facilities, "
                    "hospital hours, contact information, addresses, and operational announcements. "
                    "Use this when internal knowledge doesn't contain operational/current information. "
                    "Prefers official domains (.lk, .gov, .org)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query about hospital/healthcare operational information"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        }

def register_web_search_tool() -> WebSearchTool:
    """
    Initializes and registers the web search tool with the global ToolRegistry.

    This function acts as a factory/setup method. It instantiates the WebSearchTool, 
    creates a wrapper function (`web_search_func`) that handles the search and formatting 
    in one step, and registers this wrapper with the ToolRegistry under the name 'web_search'.
    By doing this, it links the tool's JSON definition to its actual executable Python logic,
    enabling agents to autonomously trigger web searches.

    Returns:
        WebSearchTool: The initialized instance of the WebSearchTool.
    """
    tool = WebSearchTool()
    registry = ToolRegistry()

    def web_search_func(query: str, max_results: int = 5) -> str:
        response = tool.search(query, max_results=max_results)
        return tool.format_result(response)

    registry.register_tool(
        tool_name="web_search",
        tool_function=web_search_func
    )

    return tool

_web_search_instance: Optional[WebSearchTool] = None

def get_web_search_tool() -> WebSearchTool:
    """
    Get global web search tool instance (singleton pattern).
    
    Returns:
        WebSearchTool instance
    """
    global _web_search_instance
    
    if _web_search_instance is None:
        _web_search_instance = WebSearchTool()
        
    return _web_search_instance