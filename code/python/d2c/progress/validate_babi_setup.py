from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

root = Path(__file__).parent / "babi_data"
expected = {
    "babi_train.jsonl": 18013,
    "babi_test.jsonl": 20000,
    "babi_valid.jsonl": 1987,
}

all_tasks = Counter()
all_vocab: set[str] = set()
for name, expected_count in expected.items():
    path = root / name
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    count = 0
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            record = json.loads(line)
            required = {"passage", "question", "answer", "task"}
            missing = required - record.keys()
            if missing:
                raise SystemExit(f"{name}:{line_no}: missing {sorted(missing)}")
            if not isinstance(record["task"], int) or not 1 <= record["task"] <= 20:
                raise SystemExit(f"{name}:{line_no}: invalid task")
            count += 1
            all_tasks[record["task"]] += 1
            text = f"{record['passage']} {record['question']} {record['answer']}"
            all_vocab.update(text.split())
    if count != expected_count:
        raise SystemExit(f"{name}: expected {expected_count} records, got {count}")
    print(f"{name}: {count} records, {path.stat().st_size} bytes")

print(f"tasks present: {sorted(all_tasks)}")
print(f"vocabulary estimate: {len(all_vocab)} whitespace tokens")
if set(all_tasks) != set(range(1, 21)):
    raise SystemExit("not all 20 tasks are present")
print("bAbI dataset validation: PASS")
