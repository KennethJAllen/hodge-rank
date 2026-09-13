import hashlib

import pytest

from hodge_rank import __version__
from hodge_rank.cli import main


def test_version(capsys):
    with pytest.raises(SystemExit) as exit:
        main(["--version"])
    assert exit.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_analyze_prints_results_without_changing_database(sample_db, capsys):
    before = hashlib.sha256(sample_db.read_bytes()).digest()
    main(["analyze", "--db", str(sample_db), "--city", "Seattle", "--state", "WA"])
    out = capsys.readouterr().out
    assert "5 shops, 10 comparisons" in out
    assert "Energy:" in out and "Ranking" in out and "Cafe D" in out
    assert hashlib.sha256(sample_db.read_bytes()).digest() == before


def test_empty_city_is_handled(sample_db, capsys):
    main(["analyze", "--db", str(sample_db), "--city", "Nowhere", "--state", "WA"])
    assert "No comparisons" in capsys.readouterr().out
