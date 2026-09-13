import urllib.request
import json
import time

url = "http://127.0.0.1:11434/api/generate"
payload = {
    "model": "qwen2.5-coder-32b:latest",
    "prompt": "Write a python function to check if a number is prime. Keep it short.",
    "stream": True
}

req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

print("Sending prompt to qwen2.5-coder-32b:latest...")
t0 = time.time()
first_token_time = None
tokens = []

try:
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            token = data.get("response", "")
            if token:
                if first_token_time is None:
                    first_token_time = time.time()
                    print(f"\n[Time to first token: {first_token_time - t0:.2f}s]\n")
                tokens.append(token)
                print(token, end="", flush=True)
            if data.get("done", False):
                print(f"\n\n[Done. Total time: {time.time() - t0:.2f}s, tokens: {len(tokens)}]")
except Exception as e:
    print(f"Error: {e}")
