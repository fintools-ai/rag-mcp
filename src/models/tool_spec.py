"""
Tool specification model
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass
class ToolSpec:
    """
    Standardized tool specification for RAG-MCP
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    examples: Optional[List[str]] = None
    raw_spec: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Post-initialization setup"""
        if self.tags is None:
            self.tags = []
        if self.examples is None:
            self.examples = []
    
    @classmethod
    def from_mcp_spec(cls, mcp_spec: Dict[str, Any]) -> "ToolSpec":
        """
        Create ToolSpec from MCP tool specification
        
        Args:
            mcp_spec: MCP tool specification dictionary
            
        Returns:
            ToolSpec instance
        """
        tool_info = mcp_spec.get("toolSpec", {})
        
        return cls(
            name=tool_info.get("name", ""),
            description=tool_info.get("description", ""),
            parameters=tool_info.get("inputSchema", {}).get("json", {}),
            raw_spec=mcp_spec
        )
    
    def to_mcp_spec(self) -> Dict[str, Any]:
        """
        Convert back to MCP specification format
        
        Returns:
            MCP tool specification dictionary
        """
        if self.raw_spec:
            return self.raw_spec
        
        return {
            "toolSpec": {
                "name": self.name,
                "description": self.description,
                "inputSchema": {
                    "json": self.parameters
                }
            }
        }
    
    def get_searchable_text(self) -> str:
        """
        Get text representation for embedding/search
        
        Returns:
            Combined text for semantic search
        """
        parts = []
        
        if self.name:
            parts.append(f"Tool: {self.name}")
        if self.description:
            parts.append(f"Description: {self.description}")
        if self.category:
            parts.append(f"Category: {self.category}")
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")
        
        # Add parameter descriptions
        properties = self.parameters.get("properties", {})
        for param_name, param_info in properties.items():
            param_desc = param_info.get("description", "")
            if param_desc:
                parts.append(f"Parameter {param_name}: {param_desc}")
        
        return " ".join(parts)
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the tool"""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def has_tag(self, tag: str) -> bool:
        """Check if tool has a specific tag"""
        return tag in self.tags
    
    def __repr__(self) -> str:
        return f"ToolSpec(name='{self.name}', category='{self.category}')"