from pathlib import Path

import duckdb
import pytest

from hodge_rank.ingest import ingest


def test_import_needs_only_business_and_review_files(sample_db):
    with duckdb.connect(str(sample_db), read_only=True) as con:
        assert con.execute("SHOW TABLES").fetchall() == [("business",), ("review",)]
        assert con.execute("SELECT COUNT(*) FROM business").fetchone()[0] == 6
        assert con.execute("SELECT COUNT(*) FROM review").fetchone()[0] == 24


def test_missing_input_fails_before_replacing_tables(tmp_path: Path):
    with duckdb.connect() as con:
        with pytest.raises(FileNotFoundError):
            ingest(con, tmp_path)
        assert con.execute("SHOW TABLES").fetchall() == []
