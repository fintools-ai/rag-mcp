"""
Unit tests for PrismMCPBot client integration with RAG-MCP
"""

import unittest
import json
import os
from datetime import datetime

from src.specialized.trading_agent import TradingAgentRAGMCP
from src.models.query_context import QueryContext


def load_test_tools():
    """Load tool specs for testing from prism_tool_specs.json"""
    # Load from test directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    spec_file = os.path.join(test_dir, "prism_tool_specs.json")
    
    with open(spec_file, "r") as f:
        data = json.load(f)
        tools = data.get("tools", [])
        
    if not tools:
        raise ValueError("No tools found in prism_tool_specs.json")
        
    return tools


class TestQueryEnhancement(unittest.TestCase):
    """Test query enhancement and tool selection logic"""
    
    def setUp(self):
        """Set up test environment"""
        self.tools = load_test_tools()
        self.agent = TradingAgentRAGMCP(
            ticker="SPY",
            tools=self.tools,
            similarity_threshold=0.1,  # Very low threshold to ensure both flow tools are always selected
            max_tools=5
        )
    
    def test_query_enhancement_examples(self):
        """Test specific query transformations"""
        test_cases = [
            {
                "input": "What does SPY look like? Also store the trade in memory",
                "expected_tools": ["options_order_flow_tool", "equity_order_flow_tool"]
            },
            {
                "input": "Check now and tell me what should be the trade",
                "expected_tools": ["options_order_flow_tool"]  # Don't require greeks
            },
            {
                "input": "Based on the data tell me what should be the move, what should I do",
                "expected_tools": ["financial_technical_analysis", "options_order_flow_tool"]
            }
        ]
        
        for case in test_cases:
            with self.subTest(query=case["input"]):
                context = QueryContext(
                    ticker="SPY",
                    current_time=int(datetime.now().timestamp()),
                    recommendation={},
                    conversation_history=[]
                )
                print("===")
                print(case)
                tools, enhanced_query = self.agent.get_tools_for_query(case["input"], context)
                tool_names = [t.get("toolSpec", {}).get("name") for t in tools]
                print(tool_names)
                print(enhanced_query)
                print("***")


if __name__ == "__main__":
    unittest.main(verbosity=2)