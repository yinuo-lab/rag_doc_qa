"""不依赖 Ollama 的 Agent 图测试。"""

from app.schemas.rag import RetrievedChunk
from app.services.agent_graph import DocumentAgent, ModelDecision, ToolCall
from app.services.retrieval_tool import RetrievalTool


class FakeRetriever:
    def __init__(
        self,
        score: float = 0.9,
        raises: bool = False,
        scores_by_query: dict[str, float] | None = None,
    ):
        self.score = score
        self.raises = raises
        self.scores_by_query = scores_by_query or {}

    def retrieve(self, query: str, top_k: int):
        if self.raises:
            raise RuntimeError("fake retriever failure")
        return [
            RetrievedChunk(
                source="agent_test.md",
                doc_id="agent_test.md",
                chunk_id=0,
                text="RAG uses retrieved context before generation.",
                score=self.scores_by_query.get(query, self.score),
            )
        ]


class ScriptedModel:
    def __init__(self, decisions: list[ModelDecision]):
        self.decisions = decisions
        self.messages_seen: list[list[dict]] = []

    def decide(self, messages: list[dict]) -> ModelDecision:
        self.messages_seen.append(messages)
        return self.decisions.pop(0)


def tool_decision(
    call_id: str = "call-1",
    query: str = "What is RAG?",
) -> ModelDecision:
    return ModelDecision(
        tool_call=ToolCall(
            id=call_id,
            name="retrieve_knowledge",
            arguments={"query": query, "top_k": 3},
        )
    )


def test_normal_retrieval():
    model = ScriptedModel([tool_decision(), ModelDecision(answer="RAG uses retrieved context.")])
    agent = DocumentAgent(RetrievalTool(FakeRetriever()), model)
    response = agent.ask("What is RAG?")

    assert response.answer == "RAG uses retrieved context."
    assert response.steps == 1
    assert response.stop_reason == "final_answer"
    assert len(response.sources) == 1
    assert any(message["role"] == "tool" for message in model.messages_seen[1])


def test_insufficient_context():
    model = ScriptedModel([tool_decision()])
    agent = DocumentAgent(RetrievalTool(FakeRetriever(score=0.1)), model)
    response = agent.ask("What is RAG?", max_steps=1)

    assert response.stop_reason == "insufficient_context"
    assert response.sources == []
    assert "无法回答" in response.answer
    assert len(model.messages_seen) == 1


def test_insufficient_then_rewrite_success():
    first_query = "ambiguous words"
    rewritten_query = "retrieval augmented generation"
    retriever = FakeRetriever(
        scores_by_query={first_query: 0.1, rewritten_query: 0.9}
    )
    model = ScriptedModel(
        [
            tool_decision("call-1", first_query),
            tool_decision("call-2", rewritten_query),
            ModelDecision(answer="RAG grounds generation with retrieved context."),
        ]
    )
    agent = DocumentAgent(RetrievalTool(retriever), model)
    response = agent.ask(first_query, max_steps=2)

    assert response.stop_reason == "final_answer"
    assert response.steps == 2
    assert len(response.sources) == 1
    assert len(model.messages_seen) == 3


def test_max_steps():
    model = ScriptedModel([tool_decision("call-1"), tool_decision("call-2")])
    agent = DocumentAgent(RetrievalTool(FakeRetriever()), model)
    response = agent.ask("What is RAG?", max_steps=1)

    assert response.stop_reason == "max_steps"
    assert response.steps == 1


def test_retrieval_error():
    model = ScriptedModel([tool_decision()])
    agent = DocumentAgent(RetrievalTool(FakeRetriever(raises=True)), model)
    response = agent.ask("What is RAG?")

    assert response.stop_reason == "tool_error"
    assert response.sources == []
    assert "暂时不可用" in response.answer


if __name__ == "__main__":
    test_normal_retrieval()
    test_insufficient_context()
    test_insufficient_then_rewrite_success()
    test_max_steps()
    test_retrieval_error()
    print("AGENT_GRAPH_TEST=PASS")
