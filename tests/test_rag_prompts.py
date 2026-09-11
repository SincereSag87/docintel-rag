from app.rag.prompts import INSUFFICIENT_INFORMATION_ANSWER, build_grounded_qa_messages


def test_grounded_prompt_includes_question_context_and_rules() -> None:
    messages = build_grounded_qa_messages("How many PTO days?", "[SOURCE 1]\nPTO is 15 days.")

    system = messages[0].content
    user = messages[1].content
    assert "Answer only from the supplied context" in system
    assert "Do not use prior knowledge" in system
    assert "directly states a requested value" in system
    assert "multi-part questions" in system
    assert "include every source number" in system
    assert "cite the source number of that retrieved block" in system
    assert "Do not include markdown" in system
    assert INSUFFICIENT_INFORMATION_ANSWER in system
    assert "Return valid JSON only" in system
    assert "How many PTO days?" in user
    assert "[SOURCE 1]" in user
