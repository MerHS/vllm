# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import json
import argparse
import uuid
import base64
import requests

import aiohttp
from quart import Quart, make_response, request

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
ENCODE_ENDPOINT = "http://localhost:8100/v1/chat/completions"
DECODE_ENDPOINT = "http://localhost:8200/v1/chat/completions"
CACHE_IMAGE = False

app = Quart(__name__)

# Global session for connection reuse
session = None


async def get_or_create_session():
    global session
    if session is None:
        connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=30,  # Per-host connection limit
            keepalive_timeout=30,  # Keep connections alive for 30s
            enable_cleanup_closed=True)
        session = aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT,
                                        connector=connector)
    return session


async def forward_request(url, data, req_id):
    session = await get_or_create_session()
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "x-request-id": req_id,
    }
    async with session.post(url=url, json=data, headers=headers) as response:
        if response.status == 200:
            async for chunk_bytes in response.content.iter_chunked(1024):
                yield chunk_bytes


@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    try:
        original_request_data = await request.get_json()
        req_id = str(uuid.uuid4())

        encode_request = original_request_data.copy()
        # change max_tokens = 1 to let it only do prefill
        encode_request["max_tokens"] = 1
        encode_request["kv_transfer_params"] = {"do_remote_decode": True}
        encode_request["stream"] = False
        encode_request["stream_options"] = {}

        # Capture encode response to extract kv_transfer_params
        encode_response_data = None
        async for chunk_bytes in forward_request(ENCODE_ENDPOINT,
                                                 encode_request, req_id):
            if encode_response_data is None:
                encode_response_data = b""
            encode_response_data += chunk_bytes

        # Parse the encode response to get kv_transfer_params

        encode_response_json = json.loads(encode_response_data.decode('utf-8'))
        kv_transfer_params = encode_response_json.get('kv_transfer_params', {})

        # Add kv_transfer_params to the decode request
        decode_request = original_request_data.copy()
        if kv_transfer_params:
            decode_request["kv_transfer_params"] = kv_transfer_params

        # return decode
        generator = forward_request(DECODE_ENDPOINT, decode_request, req_id)
        response = await make_response(generator)
        response.timeout = None

        return response

    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encode_port", type=int, default=8100)
    parser.add_argument("--decode_port", type=int, default=8200)
    parser.add_argument("--encode_host", type=str, default="localhost")
    parser.add_argument("--decode_host", type=str, default="localhost")
    args = parser.parse_args()

    ENCODE_ENDPOINT = f"http://{args.encode_host}:{args.encode_port}/v1/chat/completions"
    DECODE_ENDPOINT = f"http://{args.decode_host}:{args.decode_port}/v1/chat/completions"

    app.run(port=8000)
