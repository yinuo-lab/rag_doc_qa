"""最小 LangGraph Agent：模型决定是否检索，程序负责执行与停止。"""

import json
from typing import Any, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from app.schemas.agent import AgentAskResponse
from app.schemas.rag import SourceItem
from app.services.retrieval_tool import (
    RetrievalTool,
    RetrievalToolInput,
    RetrievalToolOutput,
)

RETRIEVAL_TOOL_NAME = "retrieve_knowledge"

# Function Calling 工具说明书
RETRIEVAL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": RETRIEVAL_TOOL_NAME,
        "description": "检索本地知识库。只有需要依据本地资料回答时才调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要检索的问题",
                },
                "top_k": {
                    "type": "integer",
                    "description": "最多返回的候选文本块数量，不是相似度分数",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 10,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}
#工具调用的参数
class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]

#模型这一次做出的决定
class ModelDecision(BaseModel):
    answer: str | None = None
    tool_call: ToolCall | None = None

#模型接口约定
class AgentModel(Protocol):
    def decide(self, messages: list[dict[str, Any]]) -> ModelDecision: ...

#连接 Ollama
class OpenAICompatibleAgentModel:
    """通过 Ollama 的 OpenAI 兼容接口发起真实工具调用请求。"""

    def __init__(self, model: str = "qwen3:8b"):
        self.client = OpenAI(
            base_url="http://localhost:11434/v1/",
            api_key="ollama",
        )
        self.model = model
    #让模型做决定
    def decide(self, messages: list[dict[str, Any]]) -> ModelDecision:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=[RETRIEVAL_TOOL_SCHEMA],
            tool_choice="auto",
            reasoning_effort="none",
            max_tokens=512,
        )
        message = response.choices[0].message
        #如果有工具调用的请求

        if message.tool_calls:
            call = message.tool_calls[0]
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            #返回调用的工具请求
            return ModelDecision(
                tool_call=ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=arguments,
                )
            )
        #如果没有了调用工具的请求，就返回答案
        return ModelDecision(answer=(message.content or "").strip())

#State 交接单
class AgentState(TypedDict):
    query: str
    messages: list[dict[str, Any]]#messages 保存完整对话历史
    pending_tool_call: ToolCall | None  #保存等待执行的工具请求
    tool_result: RetrievalToolOutput | None #保存工具执行结果
    sources: list[SourceItem]
    answer: str
    step: int
    max_steps: int
    stop_reason: str #保存结束原因

#DocumentAgent 和初始 State
class DocumentAgent:
    def __init__(self, retrieval_tool: RetrievalTool, model: AgentModel):
        self.retrieval_tool = retrieval_tool
        self.model = model
        self.graph = self._build_graph()  #调用 _build_graph() 创建并编译状态图。
    #ask() 是 Agent 的对外入口，接收问题和最大检索次数
    def ask(self, query: str, max_steps: int = 2) -> AgentAskResponse:
        initial_state: AgentState = {#创建初始状态
            "query": query,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是本地知识库问答助手。对于任何事实性问题，第一轮必须调用"
                        " retrieve_knowledge 检索本地资料，不能直接使用通用知识回答；"
                        "只有问候或闲聊才可以不调用工具。拿到工具结果后只能依据结果回答；"
                        "如果资料不足并且仍有检索次数，必须换关键词或缩短问题后再次检索，"
                        "不能根据常识补写答案。"
                    ),
                },
                {"role": "user", "content": query},
            ],
            "pending_tool_call": None,
            "tool_result": None,
            "sources": [],
            "answer": "",
            "step": 0,
            "max_steps": max_steps,
            "stop_reason": "",
        }
        final_state = self.graph.invoke(initial_state)#把初始 State 交给 LangGraph 开始运行，直到走到 END，返回最终 State。
        return AgentAskResponse(#构造接口响应。
            answer=final_state["answer"],
            sources=final_state["sources"],
            steps=final_state["step"],
            stop_reason=final_state["stop_reason"],
        )

    def _build_graph(self):#搭建状态图
        graph = StateGraph(AgentState)#创建使用 AgentState 的状态图
        graph.add_node("model", self._model_node)#三个节点，模型节点，工具节点，循环上限节点
        graph.add_node("tool", self._tool_node)
        graph.add_node("max_steps", self._max_steps_node)
        graph.add_edge(START, "model")#连接路线
        graph.add_conditional_edges(#使用条件边
            "model",
            self._route_after_model,
            {"tool": "tool", "max_steps": "max_steps", "end": END},
        )
        graph.add_edge("tool", "model")#工具执行结束后重新回到模型节点。这一行形成循环，也是 ReAct 的“观察结果后再次决定
        graph.add_edge("max_steps", END)#达到循环上限后直接结束。
        return graph.compile()#编译状态图，让它变成可以执行的图

    def _model_node(self, state: AgentState) -> dict[str, Any]:
        # 工具异常直接终止；
        tool_result = state["tool_result"]
        if tool_result is not None and tool_result.error:
            return {
                "answer": tool_result.message,
                "stop_reason": "tool_error",
            }
        needs_retry = tool_result is not None and not tool_result.has_enough_context#资料不足时允许模型在剩余次数内改写查询重试。
        if needs_retry and state["step"] >= state["max_steps"]:
            return {
                "answer": tool_result.message,
                "stop_reason": "insufficient_context",
            }

        try:
            decision = self.model.decide(state["messages"])#让模型根据信息做决定
        except Exception:
            return {
                "answer": "模型服务暂时不可用，请稍后再试。",
                "stop_reason": "model_error",
            }

        if decision.tool_call is not None:#如果模型做了决定
            if decision.tool_call.name != RETRIEVAL_TOOL_NAME:#如果模型请求了其他工具，拒绝执行，停止原因是 invalid_tool
                return {
                    "answer": "模型请求了未允许的工具，已停止执行。",
                    "stop_reason": "invalid_tool",
                }
            assistant_message = {#保存模型的工具请求
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": decision.tool_call.id,
                        "type": "function",
                        "function": {
                            "name": decision.tool_call.name,
                            "arguments": json.dumps(
                                decision.tool_call.arguments,
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            }
            return {#开始返回 State 更新
                "messages": [*state["messages"], assistant_message],#保留原来的消息，并在末尾追加工具请求消息
                "pending_tool_call": decision.tool_call,#把工具请求保存到 pending_tool_call
            }

        # 上一轮资料不足时，模型必须继续检索；不允许直接生成无依据答案。
        if needs_retry:
            return {
                "answer": tool_result.message,
                "stop_reason": "insufficient_context",
            }

        answer = (decision.answer or "").strip()#取出答案；如果是 None 就换成空字符串，并删除首尾空格
        if not answer:
            answer = "模型没有返回可见答案。"
        return {"answer": answer, "stop_reason": "final_answer"}#保存最终答案，并把停止原因设为 final_answer

    def _route_after_model( #条件边判断，决定下一步去哪里
        self, state: AgentState
    ) -> Literal["tool", "max_steps", "end"]:
        if state["stop_reason"]:
            return "end"
        if state["pending_tool_call"] is None:
            return "end"
        if state["step"] >= state["max_steps"]:
            return "max_steps"
        return "tool"

    def _tool_node(self, state: AgentState) -> dict[str, Any]:
        call = state["pending_tool_call"]#读取工具请求
        if call is None:
            return {
                "answer": "工具调用状态丢失，已停止执行。",
                "stop_reason": "missing_tool_call",
            }

        try:#把模型传来的参数校验成 RetrievalToolInput
            request = RetrievalToolInput.model_validate(call.arguments)
            result = self.retrieval_tool.run(request)#本地封装好的检索工具，生成需要的chunk和问题
        except ValidationError:
            result = RetrievalToolOutput(
                query=state["query"],
                chunks=[],
                sources=[],
                has_enough_context=False,
                message="模型提供的检索参数不合法。",
                error="validation_error",
            )

        tool_message = {#把检索结果转换成模型能看懂的消息
            "role": "tool",
            "tool_call_id": call.id,
            "content": result.model_dump_json(),
        }
        return {#这一段把工具执行后的结果写回 State
            "messages": [*state["messages"], tool_message],
            "pending_tool_call": None,
            "tool_result": result,
            "sources": result.sources,
            "step": state["step"] + 1,
        }

    @staticmethod #装饰器，循环保护的最后出口
    def _max_steps_node(state: AgentState) -> dict[str, Any]:
        return {
            "answer": "检索次数已达到上限，无法在限定次数内得到稳定答案。",
            "stop_reason": "max_steps",
        }
