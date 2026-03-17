import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.request_helpers import merge_headers_case_insensitive, set_header_default_case_insensitive


def test_merge_headers_case_insensitive_prefers_later_values():
    headers = merge_headers_case_insensitive(
        {"content-type": "application/json", "Authorization": "Bearer adapter"},
        {"Content-Type": "application/problem+json", "x-test": "from-passthrough"},
        {"CONTENT-TYPE": "application/xml", "authorization": "Bearer provider"},
    )

    assert [key for key in headers.keys() if key.lower() == "content-type"] == ["Content-Type"]
    assert [key for key in headers.keys() if key.lower() == "authorization"] == ["Authorization"]
    assert headers["Content-Type"] == "application/xml"
    assert headers["Authorization"] == "Bearer provider"
    assert headers["X-Test"] == "from-passthrough"



def test_set_header_default_case_insensitive_does_not_create_duplicate_content_type():
    headers = merge_headers_case_insensitive({"content-type": "application/json"})
    set_header_default_case_insensitive(headers, "Content-Type", "application/xml")

    assert [key for key in headers.keys() if key.lower() == "content-type"] == ["Content-Type"]
    assert headers["Content-Type"] == "application/json"



def test_set_header_default_case_insensitive_sets_missing_header():
    headers = merge_headers_case_insensitive({"authorization": "Bearer demo"})
    set_header_default_case_insensitive(headers, "content-type", "application/json")

    assert headers["Authorization"] == "Bearer demo"
    assert headers["Content-Type"] == "application/json"
