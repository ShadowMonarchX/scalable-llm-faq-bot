import json
import os

def load_logs(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def main():
    # Load logs and filter bad responses
    new_logs = load_logs("user_logs.jsonl")  # Each log has: {prompt, response, feedback, corrected_response}
    filtered = [log for log in new_logs if log.get("feedback") == "bad"]

    print(f"Found {len(filtered)} bad feedback entries. Adding corrected responses to fine-tuning dataset.")

    with open("dataset/fine_tune_data.jsonl", "a", encoding="utf-8") as f:
        for entry in filtered:
            if "corrected_response" in entry:
                json.dump({
                    "prompt": entry["prompt"],
                    "response": entry["corrected_response"]
                }, f)
                f.write("\n")
    print("✅ Appended new training data for fine-tuning.")

if __name__ == "__main__":
    main()
