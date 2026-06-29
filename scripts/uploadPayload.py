from openai import OpenAI
from pathlib import Path

client = OpenAI()
files = []
for file in files:
    script_directory = Path(__file__).resolve().parent
    file_path = script_directory.parent / "data" / "payloads" / file
    batch_input_file = client.files.create(
        file=open(file_path, "rb"), purpose="batch")
    print(batch_input_file)
