import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


CANONICAL_LABELS = {
    "writing_request>Write Code": ("writing_request", "Write Code"),
    "writing_request>Write English": ("writing_request", "Write English"),
    "writing_request>Code/Data Conversion": (
        "writing_request",
        "Code/Data Conversion",
    ),
    "writing_request>Summarize": ("writing_request", "Summarize"),
    "writing_request>Other": ("writing_request", "Other"),
    "editing_request>Edit Code": ("editing_request", "Edit Code"),
    "editing_request>Edit English": ("editing_request", "Edit English"),
    "editing_request>Other": ("editing_request", "Other"),
    "contextual_questions>Assignment Clarification": (
        "contextual_questions",
        "Assignment Clarification",
    ),
    "contextual_questions>Code Explanation": (
        "contextual_questions",
        "Code Explanation",
    ),
    "contextual_questions>Interpret Output": (
        "contextual_questions",
        "Interpret Output",
    ),
    "contextual_questions>Other": ("contextual_questions", "Other"),
    "conceptual_questions>Programming Language": (
        "conceptual_questions",
        "Programming Language",
    ),
    "conceptual_questions>Python Library": (
        "conceptual_questions",
        "Python Library",
    ),
    "conceptual_questions>Computer Science": (
        "conceptual_questions",
        "Computer Science",
    ),
    "conceptual_questions>Programming Tools": (
        "conceptual_questions",
        "Programming Tools",
    ),
    "conceptual_questions>Mathematics": (
        "conceptual_questions",
        "Mathematics",
    ),
    "conceptual_questions>Other Concept": (
        "conceptual_questions",
        "Other Concept",
    ),
    "verification>Verify Code": ("verification", "Verify Code"),
    "verification>Verify Report": ("verification", "Verify Report"),
    "verification>Verify Output": ("verification", "Verify Output"),
    "verification>Other": ("verification", "Other"),
    "provide_context>Assignment Information": (
        "provide_context",
        "Assignment Information",
    ),
    "provide_context>Error Message": ("provide_context", "Error Message"),
    "provide_context>Code": ("provide_context", "Code"),
    "provide_context>Other": ("provide_context", "Other"),
    "off_topic>Chit-Chat": ("off_topic", "Chit-Chat"),
    "off_topic>Greeting": ("off_topic", "Greeting"),
    "off_topic>Gratitude": ("off_topic", "Gratitude"),
    "off_topic>Other": ("off_topic", "Other"),
    "misc>Other": ("misc", "Other"),
    # Normalized into the user's taxonomy so sampling still works.
    "misc>error": ("misc", "Other"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a reproducible manual-labeling sample stratified by "
            "semester, topic, and canonical llm_label.label."
        )
    )
    parser.add_argument(
        "--input",
        default="data/outputs/studychat_with_reliance_label.jsonl",
        help="Path to the StudyChat JSONL with appended reliance labels.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=150,
        help="Number of rows to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="data/outputs/manual_labeling_sample_150.jsonl",
        help="Path to write the sampled full JSONL rows.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/outputs/manual_labeling_sample_150.csv",
        help="Path to write the flat CSV manifest for manual labeling.",
    )
    return parser.parse_args()


def canonicalize_label(raw_label: str) -> tuple[str, str]:
    canonical = CANONICAL_LABELS.get(raw_label)
    if canonical is None:
        raise ValueError(f"Unsupported llm_label.label value: {raw_label!r}")
    return canonical


def parse_reliance_label(raw_value: object) -> tuple[str | None, str | None]:
    if isinstance(raw_value, dict):
        return raw_value.get("label"), raw_value.get("reason")
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return None, None
        if isinstance(parsed, dict):
            return parsed.get("label"), parsed.get("reason")
    return None, None


def build_classifier_context(messages: object) -> tuple[list[dict], str | None, str]:
    if not isinstance(messages, list):
        return [], None, ""

    history = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") != "system"
    ]
    context_window = history[-8:] if len(history) > 8 else history

    if not context_window:
        return [], None, ""

    transcript_parts: list[str] = []
    for message in context_window[:-1]:
        role = "Student" if message.get("role") == "user" else "AI Tutor"
        transcript_parts.append(f"{role}: {message.get('content', '')}")

    target_message = context_window[-1].get("content")
    transcript_parts.append(
        "--- TARGET STUDENT MESSAGE TO CLASSIFY ---\n"
        f"Student: {target_message or ''}"
    )

    return context_window, target_message, "\n\n".join(transcript_parts)


def allocate_counts(counts: dict[str, int], target: int) -> dict[str, int]:
    total = sum(counts.values())
    if total == 0 or target <= 0:
        return {key: 0 for key in counts}

    target = min(target, total)
    ideal = {key: counts[key] * target / total for key in counts}
    allocation = {key: math.floor(ideal[key]) for key in counts}
    assigned = sum(allocation.values())

    while assigned < target:
        eligible = [key for key, count in counts.items()
                    if allocation[key] < count]
        if not eligible:
            break
        best_key = max(
            eligible,
            key=lambda key: (ideal[key] - allocation[key], counts[key], key),
        )
        allocation[best_key] += 1
        assigned += 1

    return allocation


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            llm_label = record.get("llm_label")
            if not isinstance(llm_label, dict) or "label" not in llm_label:
                raise ValueError(
                    f"Invalid llm_label at line {line_number} in {path}: {llm_label!r}"
                )

            broad_label, specific_label = canonicalize_label(
                llm_label["label"])
            reliance_label, reliance_reason = parse_reliance_label(
                record.get("reliance_label")
            )
            context_window, target_message, classifier_transcript = build_classifier_context(
                record.get("messages")
            )

            enriched_record = dict(record)
            enriched_record["_sample_metadata"] = {
                "canonical_llm_label": f"{broad_label}>{specific_label}",
                "broad_label": broad_label,
                "specific_label": specific_label,
                "reliance_label_label": reliance_label,
                "reliance_label_reason": reliance_reason,
                "classifier_context_messages": context_window,
                "classifier_context_message_count": len(context_window),
                "classifier_target_message": target_message,
                "classifier_transcript": classifier_transcript,
            }
            records.append(enriched_record)

    return records


def sample_records(records: list[dict], sample_size: int, seed: int) -> list[dict]:
    if sample_size > len(records):
        raise ValueError(
            f"Requested sample_size={sample_size}, but only {len(records)} records exist."
        )

    buckets: dict[str, dict[str, dict[str, list[dict]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for record in records:
        metadata = record["_sample_metadata"]
        semester = record["semester"]
        topic = record["topic"]
        label = metadata["canonical_llm_label"]
        buckets[semester][topic][label].append(record)

    semester_counts = {
        semester: sum(
            len(records_in_label)
            for topic_buckets in topic_buckets_by_semester.values()
            for records_in_label in topic_buckets.values()
        )
        for semester, topic_buckets_by_semester in buckets.items()
    }
    semester_allocation = allocate_counts(semester_counts, sample_size)

    randomizer = random.Random(seed)
    sampled: list[dict] = []

    for semester, semester_target in semester_allocation.items():
        if semester_target == 0:
            continue

        topic_buckets = buckets[semester]
        topic_counts = {
            topic: sum(len(records_in_label)
                       for records_in_label in label_buckets.values())
            for topic, label_buckets in topic_buckets.items()
        }
        topic_allocation = allocate_counts(topic_counts, semester_target)

        for topic, topic_target in topic_allocation.items():
            if topic_target == 0:
                continue

            label_buckets = topic_buckets[topic]
            label_counts = {
                label: len(records_in_label) for label, records_in_label in label_buckets.items()
            }
            label_allocation = allocate_counts(label_counts, topic_target)

            for label, label_target in label_allocation.items():
                if label_target == 0:
                    continue
                sampled.extend(randomizer.sample(
                    label_buckets[label], label_target))

    if len(sampled) != sample_size:
        raise ValueError(
            f"Sampling failed: expected {sample_size} rows, got {len(sampled)} rows."
        )

    randomizer.shuffle(sampled)
    return sampled


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_order",
        "chatId",
        "interactionCount",
        "semester",
        "topic",
        "canonical_llm_label",
        "broad_label",
        "specific_label",
        "llm_label_label_raw",
        "reliance_label_label",
        "reliance_label_reason",
        "classifier_context_message_count",
        "classifier_target_message",
        "classifier_transcript",
        "prompt",
        "response",
        "messages",
        "chatTitle",
        "chatStartTime",
        "chatTotalInteractionCount",
        "timestamp",
        "manual_label",
        "manual_notes",
    ]

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for index, record in enumerate(records, start=1):
            metadata = record["_sample_metadata"]
            writer.writerow(
                {
                    "sample_order": index,
                    "chatId": record["chatId"],
                    "interactionCount": record["interactionCount"],
                    "semester": record["semester"],
                    "topic": record["topic"],
                    "canonical_llm_label": metadata["canonical_llm_label"],
                    "broad_label": metadata["broad_label"],
                    "specific_label": metadata["specific_label"],
                    "llm_label_label_raw": record["llm_label"]["label"],
                    "reliance_label_label": metadata["reliance_label_label"],
                    "reliance_label_reason": metadata["reliance_label_reason"],
                    "classifier_context_message_count": metadata[
                        "classifier_context_message_count"
                    ],
                    "classifier_target_message": metadata[
                        "classifier_target_message"
                    ],
                    "classifier_transcript": metadata["classifier_transcript"],
                    "prompt": record.get("prompt"),
                    "response": record.get("response"),
                    "messages": record.get("messages"),
                    "chatTitle": record.get("chatTitle"),
                    "chatStartTime": record.get("chatStartTime"),
                    "chatTotalInteractionCount": record.get(
                        "chatTotalInteractionCount"
                    ),
                    "timestamp": record.get("timestamp"),
                    "manual_label": "",
                    "manual_notes": "",
                }
            )


def print_summary(records: list[dict], sample_size: int, seed: int) -> None:
    semester_counts = Counter(record["semester"] for record in records)
    topic_counts = Counter(
        (record["semester"], record["topic"]) for record in records)
    label_counts = Counter(
        record["_sample_metadata"]["canonical_llm_label"] for record in records
    )

    print(f"Sample size: {sample_size}")
    print(f"Seed: {seed}")
    print("Sample by semester:")
    for semester, count in sorted(semester_counts.items()):
        print(f"  {semester}: {count}")
    print("Top sample semester/topic strata:")
    for (semester, topic), count in topic_counts.most_common(10):
        print(f"  {semester} / {topic}: {count}")
    print("Top sample labels:")
    for label, count in label_counts.most_common(10):
        print(f"  {label}: {count}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_jsonl_path = Path(args.output_jsonl)
    output_csv_path = Path(args.output_csv)

    records = load_records(input_path)
    sampled_records = sample_records(records, args.sample_size, args.seed)

    write_jsonl(output_jsonl_path, sampled_records)
    write_csv(output_csv_path, sampled_records)
    print_summary(sampled_records, args.sample_size, args.seed)
    print(f"JSONL written to: {output_jsonl_path}")
    print(f"CSV written to: {output_csv_path}")


if __name__ == "__main__":
    main()
