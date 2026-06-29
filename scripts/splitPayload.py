# Failed horribly due to enqueued token limit, keep for posterity
from pathlib import Path

script_directory = Path(__file__).resolve().parent

file_path = script_directory.parent / "data" / \
    "payloads" / "batch_requests.jsonl"
LINES_PER_FILE = 8500


def split_jsonl(input_path, lines_per_chunk):
    with open(input_path, 'r', encoding='utf-8') as infile:
        file_count = 1
        line_count = 0
        output_path = script_directory.parent / "data" / \
            "payloads" / f'batch_requests_part{file_count}.jsonl'
        outfile = open(
            output_path, 'w', encoding='utf-8')

        for line in infile:
            if line_count >= lines_per_chunk:
                outfile.close()
                print(f"Saved batch_requests_part{file_count}.jsonl")
                file_count += 1
                output_path = script_directory.parent / "data" / \
                    "payloads" / f'batch_requests_part{file_count}.jsonl'
                outfile = open(
                    output_path, 'w', encoding='utf-8')
                line_count = 0

            outfile.write(line)
            line_count += 1

        outfile.close()
        print(f"Saved batch_requests_part{file_count}.jsonl")


split_jsonl(file_path, LINES_PER_FILE)
print("Splitting complete.")
