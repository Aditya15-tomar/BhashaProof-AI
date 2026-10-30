
import json
from pathlib import Path

path = Path("vectorstore/chunks.json")

with path.open("r", encoding="utf-8") as file:
    data = json.load(file)

def inspect(value):
    if isinstance(value, dict):
        filename = str(
            value.get("source")
            or value.get("filename")
            or value.get("document")
            or value.get("metadata", {}).get("source", "")
            if isinstance(value.get("metadata", {}), dict)
            else value.get("source", "")
        )

        if "sample_policy.pdf" in filename:
            print("\n" + "=" * 60)
            print("METADATA:", value.get("metadata", {}))
            print("TEXT:", value.get("text", value.get("page_content", "")))

        for child in value.values():
            if isinstance(child, (dict, list)):
                inspect(child)

    elif isinstance(value, list):
        for child in value:
            inspect(child)

inspect(data)
