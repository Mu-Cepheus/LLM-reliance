import json
import tiktoken
from pathlib import Path

script_directory = Path(__file__).resolve().parent
encoding = tiktoken.get_encoding("o200k_base")

INPUT_FILE = script_directory.parent / \
    "data" / "payloads" / "batch_requests.jsonl"
TOKEN_LIMIT_PER_FILE = 19_000_000


def split_by_token_limit():
    file_count = 1
    current_token_count = 0
    output_path = script_directory.parent / "data" / \
        "payloads" / f'batch_requests_part{file_count}.jsonl'

    outfile = open(
        output_path, 'w', encoding='utf-8')

    with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
        for line in infile:
            record = json.loads(line)

            messages_str = json.dumps(record['body']['messages'])
            input_tokens = len(encoding.encode(messages_str))

            max_output = record['body'].get('max_completion_tokens', 100)

            total_request_tokens = input_tokens + max_output

            if current_token_count + total_request_tokens > TOKEN_LIMIT_PER_FILE:
                outfile.close()
                print(
                    f"Saved Part {file_count} with {current_token_count:,} tokens.")
                file_count += 1
                outfile = open(
                    output_path, 'w', encoding='utf-8')
                current_token_count = 0

            outfile.write(line)
            current_token_count += total_request_tokens

    outfile.close()
    print(f"Saved Part {file_count} with {current_token_count:,} tokens.")
    print("Splitting Complete! Upload Part 1 to OpenAI.")


if __name__ == "__main__":
    split_by_token_limit()
