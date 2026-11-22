#!/usr/bin/env python3
"""Exercise the pinned SGLang chat API with a synthetic image, archive and tool."""

import argparse
import base64
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

MODEL = "thinkingmachines/Inkling-2-Mini"
CONTENTS = {"short": 130, "half": 7600, "near-full": 14150}
RANGES = {"short": (1, 10000), "half": (240000, 280000), "near-full": (480000, 522239)}
CODES = ["KELP-NOVA-0172", "LARK-WREN-6429", "DUNE-STAR-3186"]
BRANCH_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_branch",
        "description": "Returns the branch name for a job.",
        "parameters": {
            "type": "object",
            "properties": {"job_id": {"type": "integer"}},
            "required": ["job_id"],
            "additionalProperties": False,
        },
    },
}


def require(condition, error):
    if not condition:
        raise RuntimeError(error)


def request(endpoint, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = Request(endpoint + path, data, {"Content-Type": "application/json"})
    start = time.monotonic()
    with urlopen(req, timeout=1800) as response:
        return json.load(response), time.monotonic() - start


def chat(endpoint, messages, **settings):
    return request(endpoint, "/chat/completions", {
        "model": MODEL, "messages": messages, "temperature": 0,
        "reasoning_effort": "low", "max_tokens": 2048, **settings,
    })


def tool_round_trip(endpoint):
    messages = [{"role": "user", "content":
                 "Call lookup_branch to retrieve the branch for job 427. Then repeat the exact branch."}]
    response, first = chat(endpoint, messages, tools=[BRANCH_TOOL],
                           tool_choice={"type": "function", "function": {"name": "lookup_branch"}})
    assistant = response["choices"][0]["message"]
    calls = assistant.get("tool_calls") or []
    require(len(calls) == 1, "Expected exactly one tool call")
    call = calls[0]
    require(call["function"]["name"] == "lookup_branch" and
            json.loads(call["function"]["arguments"]) == {"job_id": 427}, "Incorrect tool arguments")
    messages += [
        {"role": "assistant", "content": assistant.get("content"), "tool_calls": [call]},
        {"role": "tool", "name": "lookup_branch", "tool_call_id": call["id"], "content": "reef-sesame"},
    ]
    response, follow_up = chat(endpoint, messages, tools=[BRANCH_TOOL], tool_choice="none")
    require("reef-sesame" in (response["choices"][0]["message"].get("content") or ""),
            "Model did not recover the tool result")
    return {"check": "tool-round-trip", "passed": True, "seconds": first + follow_up}


def fixture(size):
    count = CONTENTS[size]
    positions = {int(count * fraction): index for index, fraction in enumerate((.12, .47, .89))}
    lines = []
    for row in range(count):
        if row in positions:
            index = positions[row]
            lines.append(f"Audit value {index + 1} = {CODES[index]}.")
        status = ("approved", "pending", "released")[row % 3]
        lines.append(
            f"Log entry {row + 1}: equipment {1000 + (row * 329) % 8999} reserved on day "
            f"{1 + (row * 23) % 29} for project {19000 + (row * 773) % 69000} at minute "
            f"{480 + (row * 438) % 558}, with duration {20 + (row * 684) % 119} minutes and status {status}."
        )
    image = Path(__file__).with_name("fixtures") / "import-status.png"
    png = base64.b64encode(image.read_bytes()).decode()
    return [
        {"type": "text", "text": "\n".join(lines)},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png}},
        {"type": "text", "text": "List all three exact audit values in order, followed by the number next to "
         "failed in the image. Output only the four values."},
    ]


def retrieval(endpoint, size):
    info, seconds = chat(endpoint, [{"role": "user", "content": fixture(size)}])
    content = info["choices"][0]["message"]["content"] or ""
    actual = re.findall(r"[A-Z]{4}-[A-Z]{4}-\d{4}|\b73\b", content)
    require(actual == CODES + ["73"], "One or more archive values or the image code were incorrect")
    usage = info["usage"]
    tokens = usage["prompt_tokens"]
    image_tokens = (usage.get("prompt_tokens_details") or {}).get("image_tokens", 0)
    lower, upper = RANGES[size]
    require(lower <= tokens <= upper and image_tokens > 0, "Unexpected input token accounting")
    return {"check": size, "passed": True, "seconds": seconds, "prompt_tokens": tokens,
            "image_tokens": image_tokens}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18121/v1")
    parser.add_argument("--size", choices=CONTENTS, default="short")
    args = parser.parse_args()
    endpoint = args.base_url.rstrip("/")
    cards, _ = request(endpoint, "/models")
    require(any(m["id"] == MODEL and m.get("max_model_len") == 524288 for m in cards["data"]),
            "Expected the pinned 524,288-token Inkling service")
    print(json.dumps(tool_round_trip(endpoint)), flush=True)
    print(json.dumps(retrieval(endpoint, args.size)), flush=True)


if __name__ == "__main__":
    main()
