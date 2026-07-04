import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join merged batch output rows onto StudyChat rows by "
            "(chatId, interactionCount) and append the raw assistant "
            "message content as a new field."
        )
    )
    parser.add_argument(
        "--studychat",
        default="/workspaces/StudyChat/data.jsonl",
        help="Path to the StudyChat JSONL dataset.",
    )
    parser.add_argument(
        "--merged-output",
        default="data/outputs/merged_output.jsonl",
        help="Path to the merged output JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="data/outputs/studychat_with_reliance_label.jsonl",
        help="Path to write the joined JSONL file.",
    )
    parser.add_argument(
        "--field-name",
        default="reliance_label",
        help="Name of the new field to append to each StudyChat record.",
    )
    return parser.parse_args()


def load_studychat_index(path: Path) -> dict[tuple[str, int], dict]:
    index: dict[tuple[str, int], dict] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record["chatId"], int(record["interactionCount"]))
            if key in index:
                raise ValueError(
                    f"Duplicate StudyChat key {key} at line {line_number} in {path}"
                )
            index[key] = record
    return index


def extract_lookup_key(custom_id: str) -> tuple[str, int]:
    try:
        chat_id, interaction_count = custom_id.rsplit("_", 1)
        return chat_id, int(interaction_count)
    except ValueError as exc:
        raise ValueError(f"Invalid custom_id format: {custom_id!r}") from exc


def extract_message_content(record: dict) -> str | None:
    try:
        return record["response"]["body"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def main() -> None:
    args = parse_args()
    studychat_path = Path(args.studychat)
    merged_output_path = Path(args.merged_output)
    output_path = Path(args.output)

    studychat_index = load_studychat_index(studychat_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    matched_rows = 0
    unmatched_rows: list[tuple[int, str, int]] = []
    missing_content_rows: list[int] = []

    with merged_output_path.open() as merged_handle, output_path.open("w") as output_handle:
        for line_number, line in enumerate(merged_handle, start=1):
            if not line.strip():
                continue

            total_rows += 1
            merged_record = json.loads(line)
            chat_id, interaction_count = extract_lookup_key(
                merged_record["custom_id"])
            studychat_record = studychat_index.get(
                (chat_id, interaction_count))

            if studychat_record is None:
                unmatched_rows.append(
                    (line_number, chat_id, interaction_count))
                continue

            matched_rows += 1
            joined_record = dict(studychat_record)
            joined_record[args.field_name] = extract_message_content(
                merged_record)

            if joined_record[args.field_name] is None:
                missing_content_rows.append(line_number)

            output_handle.write(json.dumps(
                joined_record, ensure_ascii=False) + "\n")

    print(f"StudyChat rows indexed: {len(studychat_index)}")
    print(f"Merged output rows processed: {total_rows}")
    print(f"Matched rows written: {matched_rows}")
    print(f"Unmatched merged rows: {len(unmatched_rows)}")
    if unmatched_rows:
        preview = ", ".join(
            f"line {line_number} -> ({chat_id}, {interaction_count})"
            for line_number, chat_id, interaction_count in unmatched_rows[:10]
        )
        print(f"Unmatched preview: {preview}")
    print(f"Rows with missing message content: {len(missing_content_rows)}")
    if missing_content_rows:
        preview = ", ".join(str(line_number)
                            for line_number in missing_content_rows[:10])
        print(f"Missing content preview: {preview}")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
