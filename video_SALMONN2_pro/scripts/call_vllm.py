# Copyright (2026) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import sys
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a video to a running Qwen3-VL vLLM OpenAI server."
    )
    # --- connection ---
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--base-url",
        default=None,
        help="Full base URL (e.g. http://host:port/v1). Overrides --host/--port.",
    )
    parser.add_argument("--served-model-name", default="qwen3-vl")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout", type=float, default=600.0)

    # --- input (mirrors debug_vllm.py) ---
    parser.add_argument(
        "--video",
        default="/path/to/video.mp4",
        help="Local video path or URL. Local paths are sent as file:// URLs "
        "(the server must run with --allowed-local-media-path).",
    )
    parser.add_argument(
        "--prompt",
        default="\nDescribe this video in detail. ",
    )

    # --- sampling (mirrors debug_vllm.py) ---
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)

    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens as they are generated.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print the full JSON response instead of just the caption.",
    )
    return parser.parse_args()


def _video_url(video: str) -> str:
    """Turn a local path into a file:// URL; pass through real URLs."""
    if "://" in video:
        return video
    return Path(video).resolve().as_uri()


def _post(url: str, payload: dict, api_key: str, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_stream(url: str, payload: dict, api_key: str, timeout: float) -> str:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    chunks: list[str] = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            body = line[len("data:"):].strip()
            if body == "[DONE]":
                break
            event = json.loads(body)
            delta = event.get("choices", [{}])[0].get("delta", {})
            piece = delta.get("content")
            if piece:
                chunks.append(piece)
                sys.stdout.write(piece)
                sys.stdout.flush()
    sys.stdout.write("\n")
    return "".join(chunks)


def main() -> None:
    args = parse_args()

    base_url = args.base_url or f"http://{args.host}:{args.port}/v1"
    url = f"{base_url.rstrip('/')}/chat/completions"

    payload = {
        "model": args.served_model_name,
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": _video_url(args.video)},
                    },
                    {"type": "text", "text": args.prompt},
                ],
            }
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": args.stream,
    }
    if args.top_k >= 0:
        payload["top_k"] = args.top_k

    print(f"endpoint: {url}")
    print(f"video: {args.video}")
    print("caption:")

    if args.stream:
        _post_stream(url, payload, args.api_key, args.timeout)
        return

    response = _post(url, payload, args.api_key, args.timeout)
    if args.raw:
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return
    print(response["choices"][0]["message"]["content"].strip())


if __name__ == "__main__":
    main()
