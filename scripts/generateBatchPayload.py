import json
import os

filePath = os.path.abspath('../StudyChat/data.jsonl')
instructionsPath = os.path.abspath('../StudyChat/instructions.json')

with open(instructionsPath, 'r') as f:
    assignmentInstructions = json.load(f)

BASE_SYSTEM_PROMPT = """You are classifying student messages sent to an AI programming tutor.
Assign exactly one of three labels based on the student's behaviour:

THOUGHTLESS_USE — The student delegates the task entirely without engagement.
Indicators: pasting assignment instructions verbatim, submitting error tracebacks with no explanation, asking for complete solutions, single-word or one-line prompts with no context, no follow-up after receiving output.

REFLECTIVE_USE — The student engages collaboratively with the AI.
Indicators: iterating on a prior response, asking for alternatives, modifying a suggestion before accepting it, building on partial output, checking whether an approach is correct before proceeding.

CAUTIOUS_USE — The student critically evaluates or challenges the AI output.
Indicators: questioning the logic of a response, identifying an error in the output, asking for explanation of a concept rather than raw code, explicitly pushing back on a suggestion.

Here are the assignment instructions the student is working on:
"""


def generateBatchPayload(inputPath, outputPath):
    records = []

    with open(inputPath, 'r') as inputFile:
        for line in inputFile:
            records.append(json.loads(line))

    # For prompt caching, arrange the records
    records.sort(key=lambda x: f"{x['semester']}_{x['topic']}")

    with open(outputPath, 'w') as outputFile:
        for record in records:
            dictKey = f"{record['semester']}_{record['topic']}"
            specificInstruction = assignmentInstructions.get(
                dictKey, "Instructions not found.")

            fullSystemPrompt = BASE_SYSTEM_PROMPT + specificInstruction

            # Extract 4 previous interactions for context
            history = [message for message in record['messages']
                       if message['role'] != 'system']
            contextWindow = history[-8:] if len(history) > 8 else history

            # Combine messages into single text block and add specific target identifier
            chatTranscript = ""
            for message in contextWindow[:-1]:
                role = "Student" if message['role'] == 'user' else "AI Tutor"
                chatTranscript += f"{role}: {message['content']}\n\n"

            finalMessage = contextWindow[-1]['content']
            chatTranscript += f"--- TARGET STUDENT MESSAGE TO CLASSIFY ---\nStudent: {finalMessage}"

            batchItem = {
                "custom_id": f"{record['chatId']}_{record['interactionCount']}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-5.4-mini",
                    "messages": [
                        {"role": "system", "content": fullSystemPrompt},
                        {"role": "user", "content": f"Classify the target student message based on the history:\n\n{chatTranscript}"}
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "reliance_classification",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "label": {
                                        "type": "string",
                                        "description": "The reliance category of the student.",
                                        "enum": [
                                            "THOUGHTLESS_USE",
                                            "REFLECTIVE_USE",
                                            "CAUTIOUS_USE"
                                        ]
                                    },
                                    "reason": {
                                        "type": "string",
                                        "description": "A one-sentence justification for the chosen label."
                                    }
                                },
                                "required": ["label", "reason"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "max_completion_tokens": 100,
                    "temperature": 0
                }
            }
            outputFile.write(json.dumps(batchItem) + '\n')


generateBatchPayload(filePath, 'batch_requests.jsonl')
