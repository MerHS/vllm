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


async def forward_request(url, data, req_id):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "x-request-id": req_id,
        }
        async with session.post(url=url, json=data,
                                headers=headers) as response:
            if response.status == 200:
                # if response.headers.get('Transfer-Encoding') == 'chunked':
                if True:
                    async for chunk_bytes in response.content.iter_chunked(
                            1024):
                        yield chunk_bytes
                else:
                    content = await response.read()
                    yield content


def encode_image_base64_from_url(image_url: str) -> str:
    """Encode an image retrieved from a remote url to base64 format."""

    with requests.get(image_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode('utf-8')

    return result


@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    try:
        original_request_data = await request.get_json()
        req_id = str(uuid.uuid4())

        if CACHE_IMAGE:
            for messages in original_request_data.get("messages", []):
                for content in messages.get("content", []):
                    if content.get("type", "") == "image_url":
                        image_url = content["image_url"]
                        url = image_url["url"]
                        if url.startswith("http"):
                            image_url["url"] = encode_image_base64_from_url(
                                image_url=url)

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
    parser.add_argument("--cache_image", action="store_true")
    args = parser.parse_args()

    ENCODE_ENDPOINT = f"http://{args.encode_host}:{args.encode_port}/v1/chat/completions"
    DECODE_ENDPOINT = f"http://{args.decode_host}:{args.decode_port}/v1/chat/completions"
    CACHE_IMAGE = args.cache_image

    app.run(port=8000)
