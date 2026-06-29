from openai import OpenAI
client = OpenAI()

batch_input_file_ids = []

for file in batch_input_file_ids:
    batch = client.batches.create(
        input_file_id=file,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "Begin annotation job for dataset"
        }
    )
    print(batch)
