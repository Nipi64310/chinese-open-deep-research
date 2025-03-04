import os
import json
from typing import Any, Dict, List, Mapping, Optional, Union, Sequence, Type, Callable, cast
from zhipuai import ZhipuAI
from pydantic import BaseModel, Field
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.schema import ChatResult, AIMessage, HumanMessage, SystemMessage, ChatMessage
from langchain.schema.messages import BaseMessage
from langchain.schema.output import ChatGeneration
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel as TypeBaseModel
from langchain_core.language_models.chat_models import LanguageModelInput
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.utils.pydantic import is_basemodel_subclass

class ZhipuAIChat(BaseChatModel):
    """智谱AI聊天模型的LangChain集成"""
    
    client: Any = None
    model_name: str = "glm-4"
    temperature: float = 0.7
    top_p: float = 0.7
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        api_key =  "ae3de29259f2449d8cdea968ed350f11.5TdJyVRUlbXIXbuo"
        if not api_key:
            raise ValueError("ZHIPUAI_API_KEY must be provided either as an argument or as an environment variable")
        self.client = ZhipuAI(api_key=api_key)
    
    def _convert_messages_to_zhipu_format(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """将LangChain消息格式转换为智谱AI格式"""
        zhipu_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                zhipu_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                zhipu_messages.append({"role": "assistant", "content": message.content})
            elif isinstance(message, SystemMessage):
                zhipu_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, ChatMessage):
                role = message.role
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                zhipu_messages.append({"role": role, "content": message.content})
            else:
                raise ValueError(f"Message type {type(message)} not supported")
        return zhipu_messages
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """生成聊天完成"""
        zhipu_messages = self._convert_messages_to_zhipu_format(messages)
        
        params = {
            "model": self.model_name,
            "messages": zhipu_messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        
        if stop:
            params["stop"] = stop
            
        # 处理工具调用
        if "tools" in kwargs:
            params["tools"] = kwargs.pop("tools")
            
        if "tool_choice" in kwargs:
            params["tool_choice"] = kwargs.pop("tool_choice")
            
        # 添加JSON输出格式支持
        if "response_format" in kwargs:
            params["response_format"] = kwargs.pop("response_format")
            
        # 合并其他参数
        for k, v in kwargs.items():
            params[k] = v

        response = self.client.chat.completions.create(**params)
        message = response.choices[0].message
        
        # 处理可能的工具调用
        additional_kwargs = {}
        if hasattr(message, "tool_calls") and message.tool_calls:
            # 将智谱AI的工具调用格式转换为LangChain期望的格式
            converted_tool_calls = []
            for tool_call in message.tool_calls:
                # 创建符合LangChain期望的工具调用字典
                converted_tool_call = {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                converted_tool_calls.append(converted_tool_call)
            additional_kwargs["tool_calls"] = converted_tool_calls
        
        # 确保content不为None
        content = message.content if message.content is not None else ""
        
        generation = ChatGeneration(
            message=AIMessage(content=content, **additional_kwargs),
            generation_info={"finish_reason": response.choices[0].finish_reason}
        )
        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        return ChatResult(generations=[generation], llm_output={"token_usage": token_usage})
    
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """绑定工具到模型"""
        from langchain.tools.convert_to_openai import format_tool_to_openai_function
        
        # 转换工具为智谱AI格式的函数
        zhipu_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                # 如果已经是字典格式，确保符合智谱AI的格式要求
                if "type" in tool and tool["type"] == "function" and "function" in tool:
                    zhipu_tools.append(tool)
                else:
                    # 尝试转换为智谱AI格式
                    zhipu_tools.append({
                        "type": "function",
                        "function": tool
                    })
            elif isinstance(tool, type) and is_basemodel_subclass(tool):
                # 处理Pydantic模型
                schema = tool.schema()
                zhipu_tools.append({
                    "type": "function",
                    "function": {
                        "name": schema.get("title", tool.__name__),
                        "description": schema.get("description", ""),
                        "parameters": schema
                    }
                })
            elif isinstance(tool, BaseTool):
                # 从BaseTool转换为智谱AI格式
                openai_function = format_tool_to_openai_function(tool)
                zhipu_tools.append({
                    "type": "function",
                    "function": {
                        "name": openai_function["name"],
                        "description": openai_function["description"],
                        "parameters": openai_function["parameters"]
                    }
                })
            else:
                raise ValueError(f"Tool type {type(tool)} not supported")
        
        # 创建一个新的模型实例，带有工具
        new_model = self.clone()
        
        # 定义一个包装函数来处理输入
        def _invoke(input: LanguageModelInput) -> BaseMessage:
            messages = []
            if isinstance(input, str):
                messages = [HumanMessage(content=input)]
            elif isinstance(input, list):
                messages = input
            else:
                raise ValueError(f"Input type {type(input)} not supported")
            
            # 添加工具和工具选择
            tool_choice = kwargs.get("tool_choice", "auto")
            
            # 设置JSON响应格式
            response_format = {"type": "json_object"}

            # 调用模型
            result = new_model._generate(
                messages=messages,
                tools=zhipu_tools,
                tool_choice=tool_choice,
                response_format=response_format
            )
            
            # 获取生成的消息
            message = result.generations[0].message
            
            # 处理工具调用响应
            if hasattr(message, 'tool_calls') and message.tool_calls:
                # 如果有工具调用，确保返回正确格式的AIMessage
                return AIMessage(
                    content=message.content or "",  # 确保content不为None
                    tool_calls=message.tool_calls
                )
            
            print(message)
            return message
        
        # 返回可运行对象
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(_invoke)
    
    @property
    def _llm_type(self) -> str:
        """返回LLM类型"""
        return "zhipuai-chat"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """获取标识参数"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
    
    def clone(self) -> "ZhipuAIChat":
        """创建模型的副本"""
        return ZhipuAIChat(
            model_name=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            api_key=self.api_key
        )

    def with_structured_output(
        self,
        schema: Union[Dict[str, Any], Type[TypeBaseModel]],
        **kwargs: Any
    ) -> Runnable[LanguageModelInput, Any]:
        """添加结构化输出支持，与langchain官方实现兼容"""
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.runnables import RunnableLambda, RunnablePassthrough
        
        # 创建一个新的模型实例
        new_model = self.clone()
        
        # 判断schema是否为Pydantic类
        is_pydantic_schema = isinstance(schema, type) and issubclass(schema, BaseModel)
        
        # 创建JSON解析器
        if is_pydantic_schema:
            parser = JsonOutputParser(pydantic_object=schema)
        else:
            parser = JsonOutputParser()
        
        # 获取schema的JSON表示
        schema_json = None
        if is_pydantic_schema:
            schema_json = schema.schema()
        elif isinstance(schema, dict):
            schema_json = schema
        # 定义包装函数
        def _invoke(input: LanguageModelInput) -> Any:
            messages = []
            if isinstance(input, str):
                messages = [HumanMessage(content=input)]
            elif isinstance(input, list):
                messages = input
            else:
                raise ValueError(f"Input type {type(input)} not supported")
            
            # 添加更明确的JSON格式说明
            if schema_json:
                for i, message in enumerate(messages):
                    if isinstance(message, SystemMessage):
                        import json
                        schema_str = json.dumps(schema_json, ensure_ascii=False, indent=2)
                        format_instruction = f"""
请严格按照以下JSON格式返回结果：
{schema_str}

不要包含任何其他解释，只返回符合上述格式的JSON。
"""
                        messages[i] = SystemMessage(content=message.content + format_instruction)
                        break
            
            # 设置JSON响应格式
            response_format = {"type": "json_object"}

            print(messages)
            # 调用模型
            result = new_model._generate(
                messages=messages,
                response_format=response_format
            )
            
            # 获取模型输出的内容
            content = result.generations[0].message.content
            print(content)
            # 使用解析器解析JSON内容
            try:
                # 处理可能的格式问题
                cleaned_content = content
                if cleaned_content.startswith("```json"):
                    cleaned_content = cleaned_content.split("```json")[1]
                    if "```" in cleaned_content:
                        cleaned_content = cleaned_content.split("```")[0]
                elif "```" in cleaned_content:
                    parts = cleaned_content.split("```")
                    for part in parts:
                        if part.strip().startswith("{") or part.strip().startswith("["):
                            cleaned_content = part.strip()
                            break
                
                cleaned_content = cleaned_content.strip()
                
                # 解析JSON
                parsed_content = parser.parse(cleaned_content)
                
                # 确保返回的是正确的Pydantic对象
                if is_pydantic_schema:
                    if not isinstance(parsed_content, schema):
                        if isinstance(parsed_content, dict):
                            # 处理特殊情况：Queries类但返回格式不匹配
                            if schema.__name__ == "Queries" and "queries" not in parsed_content:
                                # 处理 {query1: "...", query2: "...", ...} 格式
                                queries = []
                                for key, value in parsed_content.items():
                                    if key.startswith("query") and isinstance(value, str):
                                        queries.append({"search_query": value})
                                
                                if queries:
                                    return schema(queries=[SearchQuery(**q) for q in queries])
                            
                            # 常规情况：尝试直接转换为Pydantic对象
                            try:
                                return schema(**parsed_content)
                            except Exception:
                                # 如果直接转换失败，可能需要进一步处理
                                pass
                
                return parsed_content
            except Exception as e:
                print(f"解析错误: {e}")
                print(f"原始内容: {content}")
                
                # 尝试手动解析不同格式
                try:
                    import json
                    # 如果是字符串，尝试直接解析JSON
                    if isinstance(content, str):
                        # 尝试清理内容
                        content_to_parse = content
                        if "```" in content_to_parse:
                            for part in content_to_parse.split("```"):
                                if part.strip().startswith("{") or part.strip().startswith("["):
                                    content_to_parse = part.strip()
                                    break
                        
                        data = json.loads(content_to_parse)
                        
                        # 如果schema是Queries类，但返回的是不同格式
                        if is_pydantic_schema and schema.__name__ == "Queries" and "queries" not in data:
                            # 处理 {query1: "...", query2: "...", ...} 格式
                            queries = []
                            for key, value in data.items():
                                if key.startswith("query") and isinstance(value, str):
                                    queries.append(SearchQuery(search_query=value))
                            
                            if queries:
                                return schema(queries=queries)
                        
                        # 如果是Pydantic模型，尝试转换
                        if is_pydantic_schema:
                            try:
                                return schema(**data)
                            except Exception:
                                # 如果转换失败，可能需要进一步处理
                                pass
                        
                        return data
                except Exception as inner_e:
                    print(f"二次解析错误: {inner_e}")
                    
                    # 最后尝试：如果是Queries类，尝试从原始内容中提取查询
                    if is_pydantic_schema and schema.__name__ == "Queries":
                        try:
                            import re
                            # 尝试从内容中提取查询
                            queries = []
                            # 查找形如 "query1": "text" 或 "search_query": "text" 的模式
                            pattern = r'["\'](query\d+|search_query)["\']\s*:\s*["\']([^"\']+)["\']'
                            matches = re.findall(pattern, content)
                            
                            for _, query_text in matches:
                                queries.append(SearchQuery(search_query=query_text))
                            
                            if queries:
                                return schema(queries=queries)
                        except Exception:
                            pass
                
                return None
        
        # 返回可运行对象
        return RunnableLambda(_invoke)

# 自定义初始化函数
def init_chat_model(provider=None, **kwargs):
    """初始化聊天模型"""
    if provider == "zhipuai":
        return ZhipuAIChat(**kwargs)
    
    # 如果不是智谱AI，则尝试使用原始的init_chat_model
    try:
        from langchain.chat_models import init_chat_model as original_init_chat_model
        return original_init_chat_model(provider=provider, **kwargs)
    except ImportError:
        raise ValueError(f"Provider {provider} not supported or langchain.chat_models.init_chat_model not available")
