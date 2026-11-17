# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Optional, Tuple
import re
import json
import time
import os
import sys
import threading
import cv2  # type: ignore
import base64

import requests  # type: ignore
from openai import OpenAI  # type: ignore
from dotenv import load_dotenv  # type: ignore

dot_file = os.path.join(os.path.dirname(__file__), "../.env")
load_dotenv(dot_file)

_openai_clients: Dict[Tuple[Optional[str], str], OpenAI] = {}
_openai_client_lock = threading.Lock()


def get_openai_client(
    base_url: Optional[str] = None, api_key: Optional[str] = None
) -> OpenAI:
    """Returns a shared OpenAI(-compatible) client, creating it on first use.

    The client is created lazily so that OPENAI_API_KEY is only required when
    an OpenAI model is actually called (e.g. Gemini-only setups don't need it).
    Passing ``base_url``/``api_key`` returns a client for a third-party
    OpenAI-compatible endpoint.
    """
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "OPENAI_API_KEY (or an api_key in model_config.yaml) must be set "
            "to use OpenAI-compatible models."
        )

    cache_key = (base_url, resolved_api_key)
    with _openai_client_lock:
        client = _openai_clients.get(cache_key)
        if client is None:
            if base_url:
                client = OpenAI(api_key=resolved_api_key, base_url=base_url)
            else:
                client = OpenAI(api_key=resolved_api_key)
            _openai_clients[cache_key] = client
    return client


def parse_openai_json_results(response: str):
    if not response or not isinstance(response, str):
        return None

    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, response, re.DOTALL)

    if not matches:
        json_pattern = r"(\{.*\}|\[.*\])"
        matches = re.findall(json_pattern, response, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            try:
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                json_string_clean = re.sub(r",\s*([\]}])", r"\1", json_string_clean)
                json_string_clean = re.sub(
                    r'\\(?![\\"/bfnrtu])', r"\\\\", json_string_clean
                )

                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue

    try:
        start_idx = response.find("{")
        end_idx = response.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_string = response[start_idx : end_idx + 1]
            json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
            json_string_clean = re.sub(r",\s*([\]}])", r"\1", json_string_clean)
            json_string_clean = re.sub(
                r'\\(?![\\"/bfnrtu])', r"\\\\", json_string_clean
            )
            return json.loads(json_string_clean)
    except Exception:
        pass

    try:
        start_idx = response.find("[")
        end_idx = response.rfind("]")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_string = response[start_idx : end_idx + 1]
            json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
            json_string_clean = re.sub(r",\s*([\]}])", r"\1", json_string_clean)
            json_string_clean = re.sub(
                r'\\(?![\\"/bfnrtu])', r"\\\\", json_string_clean
            )
            return json.loads(json_string_clean)
    except Exception:
        pass

    return None  # No valid JSON found


def call_openai_models_with_content(
    content: list[Any],
    model_name: str,
    max_retries: int = 5,
    base_interval_sec: int = 5,
    result_parsing_func: callable = parse_openai_json_results,
    generation_configs: dict = {},
    check_parsed_response_not_none: bool = True,
    system_prompt: str = None,
    client: Optional[OpenAI] = None,
):
    raw_response = None
    parsed_response = None
    openai_client = client or get_openai_client()

    for attempt in range(max_retries):
        try:
            messages = []
            if system_prompt:
                role = "developer" if model_name.startswith(("o1", "o3")) else "system"
                messages.append({"role": role, "content": system_prompt})

            messages.append({"role": "user", "content": content})

            response = openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                **generation_configs,
            )
            raw_response = response.choices[0].message.content
            parsed_response = result_parsing_func(raw_response)

            if check_parsed_response_not_none:
                assert (
                    parsed_response is not None
                ), f"Parsed response is None, Here is the raw response:\n{raw_response}"

            return {
                "raw_response": raw_response,
                "parsed_response": parsed_response,
            }

        except Exception as e:
            print(
                f"Warning: OpenAI API call (model={model_name}) failed on attempt {attempt + 1} of {max_retries}. Error: {e}",
            )

            if attempt + 1 == max_retries:
                print(f"Error: All {max_retries} retry attempts failed.")
                raise e

            delay = base_interval_sec * (2 ** (attempt - 1)) if attempt > 0 else 0
            print(f"Retrying in {delay} seconds...")
            if delay > 0:
                time.sleep(delay)

    print("Error: Exited retry loop unexpectedly.")
    return {
        "raw_response": raw_response,
        "parsed_response": parsed_response,
    }


def call_openai_models_with_text_prompt(
    prompt: str,
    model_name: str,
    result_parsing_func: callable = parse_openai_json_results,
    generation_configs: dict = {},
    check_parsed_response_not_none: bool = True,
    max_retries: int = 5,
    base_interval_sec: int = 5,
    system_prompt: str = None,
    client: Optional[OpenAI] = None,
):
    return call_openai_models_with_content(
        content=[{"type": "text", "text": prompt}],
        model_name=model_name,
        result_parsing_func=result_parsing_func,
        generation_configs=generation_configs,
        check_parsed_response_not_none=check_parsed_response_not_none,
        max_retries=max_retries,
        base_interval_sec=base_interval_sec,
        system_prompt=system_prompt,
        client=client,
    )


_IMAGE_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


def _guess_image_mime_type(data: bytes, image_path: str = "") -> str:
    """Detects an image MIME type from magic bytes, then the file extension."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    return _IMAGE_MIME_BY_EXT.get(
        os.path.splitext(image_path)[1].lower(), "image/jpeg"
    )


def get_openai_image_part_from_path(image_path: str):
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = _guess_image_mime_type(image_bytes, image_path)

    image_content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}",
            },
        }
    ]
    return image_content


def _aspect_ratio_to_image_size(aspect_ratio: str) -> str:
    """Maps a Gemini-style aspect ratio to an OpenAI-compatible image size."""
    try:
        width_str, height_str = str(aspect_ratio).split(":")
        ratio = float(width_str) / float(height_str)
    except (ValueError, ZeroDivisionError):
        return "1024x1024"

    if ratio >= 1.2:
        return "1536x1024"
    if ratio <= 0.83:
        return "1024x1536"
    return "1024x1024"


def generate_image_with_openai_compatible(
    model_name: str,
    prompt: str,
    aspect_ratio: str = "16:9",
    generation_configs: dict = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_retries: int = 5,
    base_interval_sec: int = 5,
    save_path: str = None,
) -> Optional[str]:
    """Generates an image through an OpenAI-compatible Images API.

    Returns the image as a base64 string, mirroring
    ``gemini_utils.generate_image_with_gemini``.
    """
    client = get_openai_client(base_url=base_url, api_key=api_key)

    if generation_configs:
        system_instruction = generation_configs.get("system_instruction")
        if system_instruction:
            prompt = f"{system_instruction}\n\n{prompt}"

    image_size = _aspect_ratio_to_image_size(aspect_ratio)
    # Some providers reject response_format or non-square sizes, so fall back
    # to progressively simpler payloads.
    request_variants = [
        {"size": image_size, "response_format": "b64_json"},
        {"size": image_size},
        {},
    ]

    for attempt in range(max_retries):
        last_error = None
        for extra_params in request_variants:
            try:
                response = client.images.generate(
                    model=model_name,
                    prompt=prompt,
                    **extra_params,
                )

                if not response or not getattr(response, "data", None):
                    raise ValueError("Empty image response from the API.")

                image_data = response.data[0]
                img_base64 = getattr(image_data, "b64_json", None)
                if not img_base64:
                    image_url = getattr(image_data, "url", None)
                    if not image_url:
                        raise ValueError(
                            "Image response contained neither b64_json nor url."
                        )
                    download = requests.get(image_url, timeout=120)
                    download.raise_for_status()
                    img_base64 = base64.b64encode(download.content).decode("utf-8")

                if save_path:
                    try:
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        with open(save_path, "wb") as fh:
                            fh.write(base64.b64decode(img_base64))
                        print(f"Image for prompt '{prompt}' saved to {save_path}")
                    except Exception as save_error:
                        print(
                            f"Warning: Failed to save image to {save_path}: "
                            f"{save_error}"
                        )

                return img_base64
            except Exception as e:
                last_error = e
                print(
                    f"Warning: OpenAI-compatible image generation "
                    f"(model={model_name}, params={extra_params}) failed: {e}"
                )

        if attempt + 1 == max_retries:
            print(
                f"Error: All {max_retries} retry attempts failed for image "
                f"generation."
            )
            if last_error is not None:
                print(f"Last error: {last_error}")
            break

        delay = base_interval_sec * (2 ** (attempt - 1)) if attempt > 0 else 0
        print(f"Retrying in {delay} seconds...")
        if delay > 0:
            time.sleep(delay)

    return None