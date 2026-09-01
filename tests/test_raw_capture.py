import json

from kalshi_research.storage.raw_capture import RawJsonlCapture, RawRecord, verify_hash_chain


def test_raw_capture_hash_chain_detects_tamper(tmp_path):
    capture = RawJsonlCapture(tmp_path)
    path, _ = capture.append(RawRecord("kalshi", 1_700_000_000_000_000_000, "c1", {"x": 1}))
    capture.append(RawRecord("kalshi", 1_700_000_001_000_000_000, "c1", {"x": 2}))
    assert verify_hash_chain(path)
    rows = path.read_text().splitlines()
    row = json.loads(rows[0])
    row["payload"]["x"] = 999
    rows[0] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(rows) + "\n")
    assert not verify_hash_chain(path)
