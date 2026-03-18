import logging
import uuid

from core.log_config import get_backend_log_entries


def test_backend_log_entries_include_level_and_stream_for_logger_records():
    baseline_id = get_backend_log_entries(limit=1)["max_id"]
    test_logger = logging.getLogger("Zoaholic.test.backend_logs")

    info_message = f"backend-log-info-{uuid.uuid4().hex}"
    error_message = f"backend-log-error-{uuid.uuid4().hex}"

    test_logger.info(info_message)
    test_logger.error(error_message)

    snapshot = get_backend_log_entries(since_id=baseline_id, limit=20)
    items = snapshot["items"]

    info_entry = next(item for item in items if item["message"] == info_message)
    error_entry = next(item for item in items if item["message"] == error_message)

    assert info_entry["level"] == "INFO"
    assert info_entry["stream"] == "stdout"
    assert info_entry["source"] == "logger"
    assert info_entry["logger_name"] == "Zoaholic.test.backend_logs"

    assert error_entry["level"] == "ERROR"
    assert error_entry["stream"] == "stderr"
    assert error_entry["source"] == "logger"
    assert error_entry["logger_name"] == "Zoaholic.test.backend_logs"

    filtered = get_backend_log_entries(since_id=baseline_id, limit=20, level="ERROR")
    assert any(item["message"] == error_message for item in filtered["items"])
    assert all(item["level"] == "ERROR" for item in filtered["items"])

    logger_filtered = get_backend_log_entries(
        since_id=baseline_id, limit=20, logger_name="Zoaholic.test.backend_logs"
    )
    assert {item["message"] for item in logger_filtered["items"]} == {info_message, error_message}
    assert all(item["logger_name"] == "Zoaholic.test.backend_logs" for item in logger_filtered["items"])

    errors_filtered = get_backend_log_entries(
        since_id=baseline_id,
        limit=20,
        level_group="errors",
    )
    assert any(item["message"] == error_message for item in errors_filtered["items"])
    assert all(item["level"] in {"ERROR", "CRITICAL"} for item in errors_filtered["items"])
