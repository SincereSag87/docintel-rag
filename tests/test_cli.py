from app.main import build_parser, run_clear_index


def test_cli_parses_index() -> None:
    args = build_parser().parse_args(["--index", "handbook.txt"])

    assert args.index == "handbook.txt"


def test_cli_parses_search_and_top_k() -> None:
    args = build_parser().parse_args(["--search", "pto days", "--top-k", "3"])

    assert args.search == "pto days"
    assert args.top_k == 3


def test_cli_parses_index_stats() -> None:
    args = build_parser().parse_args(["--index-stats"])

    assert args.index_stats is True


def test_cli_requires_confirmation_for_clear_index(capsys) -> None:
    exit_code = run_clear_index(confirmed=False)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "--clear-index --yes" in captured.out
