import json
from openai import OpenAI
from pathlib import Path


script_directory = Path(__file__).resolve().parent
file_path = script_directory.parent / "data" / "payloads" / "failed_request.json"
client = OpenAI()

with open(file_path, "r", encoding="utf-8") as f:
    failed_req = json.loads(f.read().strip())

payload = failed_req["body"]

print(f"Sending request for ID: {failed_req['custom_id']}...")

try:
    response = client.chat.completions.create(**payload)

    llm_output_string = response.choices[0].message.content

    final_classification = json.loads(llm_output_string)

    print("\nSUCCESS! Here is the classification:")
    print(json.dumps(final_classification, indent=4))

    batch_result_format = {
        "id": f"req_{failed_req['custom_id']}",
        "custom_id": failed_req["custom_id"],
        "response": {
            "status_code": 200,
            "request_id": response.id,
            "body": response.model_dump()
        },
        "error": None
    }

    output_path = script_directory.parent/"data" / \
        "LLM_Labeled"/"recovered_result.jsonl"
    with open(output_path, "w", encoding="utf-8") as out:
        out.write(json.dumps(batch_result_format) + "\n")

    print("\nSaved result to recovered_result.jsonl.")
    print("Append it using: cat recovered_result.jsonl >> batch_results_part1.jsonl")

except Exception as e:
    print(f"\nFAILED. Error Details:")
    print(e)
