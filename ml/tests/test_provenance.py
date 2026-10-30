from datetime import timedelta

import pytest

from src.research.provenance import audit_archive_versions


def test_provenance_flags_late_and_unknown_file_versions():
    files = [
        {"date": "2024-01-01", "source_last_modified": "Tue, 02 Jan 2024 06:00:00 GMT"},
        {"date": "2024-01-02", "source_last_modified": "Sat, 06 Jan 2024 00:01:00 GMT"},
        {"date": "2024-01-03", "source_last_modified": None},
    ]
    report = audit_archive_versions(files, timedelta(hours=48))
    assert report["files"] == 3
    assert report["modified_after_last_event_plus_delay"] == 1
    assert report["last_modified_unknown"] == 1
    assert report["point_in_time_verified"] is False
    with pytest.raises(ValueError, match="demora"):
        audit_archive_versions(files, timedelta(hours=-1))
