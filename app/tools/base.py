#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MCP 工具基类和注册系统

Model Context Protocol (MCP) 工具允许 Agent 调用外部服务获取实时数据。
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None


@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    
    def to_openai_function(self) -> Dict[str, Any]:
        """转换为 OpenAI Function Calling 格式"""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


class MCPTool(ABC):
    """MCP 工具基类"""
    
    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """返回工具定义"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """执行工具并返回结果"""
        pass
    
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """允许直接调用工具"""
        return self.execute(**kwargs)


class ToolRegistry:
    """工具注册中心"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools: Dict[str, MCPTool] = {}
        return cls._instance
    
    def register(self, tool: MCPTool) -> None:
        """注册工具"""
        self._tools[tool.definition.name] = tool
        print(f"[MCP] Registered tool: {tool.definition.name}")
    
    def get(self, name: str) -> Optional[MCPTool]:
        """获取工具"""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """列出所有工具名称"""
        return list(self._tools.keys())
    
    def get_all_definitions(self) -> List[ToolDefinition]:
        """获取所有工具定义"""
        return [tool.definition for tool in self._tools.values()]
    
    def get_openai_functions(self) -> List[Dict[str, Any]]:
        """获取所有工具的 OpenAI Function 格式"""
        return [tool.definition.to_openai_function() for tool in self._tools.values()]
    
    def execute(self, name: str, **kwargs) -> Dict[str, Any]:
        """执行指定工具"""
        tool = self.get(name)
        if not tool:
            return {"error": f"Tool '{name}' not found"}
        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return {"error": str(e)}


# 全局工具注册器
tool_registry = ToolRegistry()


def register_tool(tool_class):
    """装饰器：自动注册工具"""
    tool_instance = tool_class()
    tool_registry.register(tool_instance)
    return tool_class
