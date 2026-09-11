import json

from app.evaluation.models import RAGEvaluationReport


def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def format_evaluation_report(report: RAGEvaluationReport) -> str:
    lines = [
        f"Dataset: {report.dataset_name}",
        f"Model: {report.model}",
        f"Mode: {'retrieval-only' if report.retrieval_only else 'full RAG'}",
        "",
    ]
    for result in report.case_results:
        answer_status = "SKIP" if report.retrieval_only else _status(result.answer_correct)
        citation_status = "SKIP" if report.retrieval_only else _status(result.citation_correct)
        lines.extend(
            [
                f"Case: {result.case_id}",
                f"Question: {result.question}",
                f"Retrieval: {_status(result.retrieval_hit or not result.answerable)}",
                f"Answer: {answer_status}",
                f"Citations: {citation_status}",
                f"Latency: {result.latency_ms / 1000:.2f}s",
            ]
        )
        if not result.success:
            lines.extend(
                [
                    "Failure Analysis:",
                    f"  Expected documents: {result.expected_documents}",
                    f"  Retrieved documents: {result.retrieved_documents}",
                    f"  Cited documents: {result.cited_documents}",
                    f"  Answer: {result.answer or '<none>'}",
                    f"  Error: {result.error or '<none>'}",
                ]
            )
        lines.append("")

    lines.extend(
        [
            "Summary",
            f"Cases: {report.cases_total}",
            f"Passed: {report.cases_passed}",
            f"Retrieval Hit@K: {report.retrieval_hit_rate:.1%}",
            f"Mean Retrieval Recall@K: {report.mean_retrieval_recall:.1%}",
            f"MRR: {report.mrr:.3f}",
            f"Answer Accuracy: {report.answer_accuracy:.1%}",
            f"Citation Precision: {report.average_citation_precision:.1%}",
            f"Citation Recall: {report.average_citation_recall:.1%}",
            f"Citation Accuracy: {report.citation_accuracy:.1%}",
            f"Unknown Accuracy: {report.unknown_answer_accuracy:.1%}",
            f"Parse Success: {report.parse_success_rate:.1%}",
            f"Avg Latency: {report.average_latency_ms / 1000:.2f}s",
            f"Failed Cases: {report.failed_case_count}",
        ]
    )
    return "\n".join(lines)


def evaluation_report_to_json(report: RAGEvaluationReport) -> str:
    return json.dumps(report.model_dump(), indent=2)
