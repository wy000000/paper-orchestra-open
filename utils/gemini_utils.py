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

import re
import json
from typing import Dict, Any

from google import genai
from google.genai import types  # type: ignore
import time
import os
import mimetypes
import base64
import threading

import dotenv

from utils import model_config

dot_file = os.path.join(os.path.dirname(__file__), "../.env")
dotenv.load_dotenv(dot_file)

_genai_client = None
_genai_client_lock = threading.Lock()
_custom_genai_clients = {}


def get_genai_client(
    endpoint: "model_config.ModelEndpoint | None" = None,
) -> genai.Client:
    """Returns a shared Gemini client, creating it on first use.

    The client is created lazily so that Gemini credentials are only required
    when a Gemini model is actually called (e.g. OpenAI-only setups don't need
    them). Vertex AI takes precedence over GEMINI_API_KEY when both are set.

    When ``endpoint`` is provided, a client for that third-party
    Gemini-compatible base URL is returned instead.
    """
    if endpoint is not None:
        cache_key = (endpoint.base_url, endpoint.api_key)
        with _genai_client_lock:
            client = _custom_genai_clients.get(cache_key)
            if client is None:
                print(
                    f"Using third-party Gemini-compatible endpoint: "
                    f"{endpoint.base_url}"
                )
                client = genai.Client(
                    api_key=endpoint.api_key,
                    http_options=types.HttpOptions(base_url=endpoint.base_url),
                )
                _custom_genai_clients[cache_key] = client
        return client

    global _genai_client
    if _genai_client is None:
        with _genai_client_lock:
            if _genai_client is None:
                vertex_ai_project = os.getenv("VERTEX_AI_PROJECT")
                vertex_ai_location = os.getenv("VERTEX_AI_LOCATION")
                gemini_api_key = os.getenv("GEMINI_API_KEY")

                if vertex_ai_project and vertex_ai_location:
                    print(
                        f"Using Vertex AI configuration. Project: {vertex_ai_project}, Location: {vertex_ai_location}"
                    )
                    _genai_client = genai.Client(
                        vertexai=True,
                        project=vertex_ai_project,
                        location=vertex_ai_location,
                    )
                elif gemini_api_key:
                    print("Using Gemini API Key configuration.")
                    _genai_client = genai.Client(api_key=gemini_api_key)
                else:
                    raise ValueError(
                        "Either VERTEX_AI_PROJECT/VERTEX_AI_LOCATION or GEMINI_API_KEY"
                        " must be set to use Gemini models."
                    )
    return _genai_client


def _contents_to_openai_content(contents) -> list:
    """Converts Gemini ``types.Part``/string contents to OpenAI content parts."""
    openai_content = []
    for item in _iter_content_items(contents):
        if isinstance(item, str):
            openai_content.append({"type": "text", "text": item})
            continue

        text = getattr(item, "text", None)
        if text:
            openai_content.append({"type": "text", "text": text})
            continue

        inline_data = getattr(item, "inline_data", None)
        if inline_data is not None and getattr(inline_data, "data", None):
            mime_type = getattr(inline_data, "mime_type", None) or "image/png"
            if mime_type == "application/pdf":
                raise ValueError(
                    "PDF inputs are not supported by OpenAI-compatible chat "
                    "completions. Use call_llm_with_pdf instead."
                )
            data = inline_data.data
            if isinstance(data, str):
                data = data.encode("utf-8")
            encoded = base64.b64encode(data).decode("utf-8")
            openai_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                }
            )
            continue

        if getattr(item, "file_data", None) is not None:
            raise ValueError(
                "File inputs (file_data) are not supported by OpenAI-compatible "
                "chat completions. Use call_llm_with_pdf instead."
            )

        raise ValueError(f"Unsupported content item for OpenAI backend: {item!r}")

    return openai_content


def _iter_content_items(contents):
    if isinstance(contents, (str, bytes)):
        if isinstance(contents, bytes):
            raise ValueError("Raw bytes are not supported as LLM contents.")
        yield contents
        return

    if isinstance(contents, (list, tuple)):
        for item in contents:
            yield from _iter_content_items(item)
        return

    parts = getattr(contents, "parts", None)
    if parts:
        for part in parts:
            yield from _iter_content_items(part)
        return

    yield contents


def _split_generation_configs(generation_configs: dict):
    """Splits Gemini generation configs into OpenAI-compatible ones + system prompt."""
    configs = dict(generation_configs or {})
    system_prompt = configs.pop("system_instruction", None)

    openai_configs = {}
    for key, value in configs.items():
        if key == "max_output_tokens":
            openai_configs["max_tokens"] = value
        elif key == "stop_sequences":
            openai_configs["stop"] = value
        elif key in {
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "seed",
            "response_format",
            "logit_bias",
            "n",
            "max_tokens",
            "stop",
        }:
            openai_configs[key] = value
        # Gemini-specific options (tools, response_modalities, image_config,
        # candidate_count, safety_settings, ...) are dropped here.

    return openai_configs, system_prompt


_tool_fallback_warned_roles = set()


def _warn_tool_fallback(role: str) -> None:
    if role in _tool_fallback_warned_roles:
        return
    _tool_fallback_warned_roles.add(role)
    print(
        f"WARNING: Role '{role}' uses an OpenAI-compatible endpoint, but this "
        "call requires Google Search grounding. Falling back to native Gemini "
        "for this call."
    )


def _call_openai_compatible_with_contents(
    contents,
    endpoint: "model_config.ModelEndpoint",
    result_parsing_func,
    generation_configs: dict,
    check_parsed_response_not_none: bool,
):
    # Imported lazily to avoid pulling in the OpenAI/OpenCV stack for
    # Gemini-only runs.
    from utils.openai_utils import (
        call_openai_models_with_content,
        get_openai_client,
    )

    openai_content = _contents_to_openai_content(contents)
    # Text-only requests are sent as a plain string for better compatibility
    # with OpenAI-compatible relays.
    if openai_content and all(
        part.get("type") == "text" for part in openai_content
    ):
        openai_content = "\n\n".join(part["text"] for part in openai_content)

    openai_configs, system_prompt = _split_generation_configs(generation_configs)

    return call_openai_models_with_content(
        content=openai_content,
        model_name=endpoint.model_name,
        generation_configs=openai_configs,
        result_parsing_func=result_parsing_func,
        check_parsed_response_not_none=check_parsed_response_not_none,
        system_prompt=system_prompt,
        client=get_openai_client(
            base_url=endpoint.base_url, api_key=endpoint.api_key
        ),
    )


def parse_gemini_json_results(response: str):
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

    return None  # No valid JSON found


def parse_gemini_latex_results(response: str):
    match = re.search(r"```latex\s*(.*?)\s*```", response.strip(), re.DOTALL)
    if match:
        latex_string = match.group(1)
    else:
        latex_string = response
    return latex_string


def parse_gemini_json_latex_response(response_text: str) -> Dict[str, Any]:
    """
    Extracts the JSON worklog and the LaTeX code from the LLM response.
    Expected format:
      ... arbitrary text ...
      ```json
      { ... }
      ```
      ... arbitrary text ...
      ```latex
      ...
      ```
    """
    result = {"json": None, "latex": None}

    json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
    if json_match:
        try:
            result["json"] = json.loads(json_match.group(1))
        except json.JSONDecodeError:
            print(" >> Error: Failed to parse JSON worklog from response.")
            result["json"] = {"error": "Invalid JSON format returned"}

    latex_match = re.search(r"```latex\s*(.*?)\s*```", response_text, re.DOTALL)
    if latex_match:
        result["latex"] = latex_match.group(1)
    else:
        print(" >> Warning: No ```latex``` block found.")

    return json.dumps(result)


def call_gemini_with_contents(
    contents: types.Content,
    model_name: str,
    model_role: str = model_config.MAIN_ROLE,
    max_retries: int = 5,
    base_interval_sec: int = 5,
    result_parsing_func: callable = parse_gemini_json_results,
    generation_configs: dict = {},
    check_parsed_response_not_none: bool = True,
):
    """
    Calls the Gemini API (text-only) with retry logic and exponential backoff.

    If a third-party endpoint is enabled for ``model_role``, the call is routed
    there instead. OpenAI-compatible endpoints fall back to native Gemini for
    calls that require Gemini-only tools (e.g. Google Search grounding).
    """

    raw_response = None
    parsed_response = None

    generation_configs = generation_configs or {}

    endpoint = model_config.get_endpoint(model_role)
    if endpoint is not None:
        if endpoint.api_style == "openai":
            if generation_configs.get("tools"):
                _warn_tool_fallback(model_role)
                endpoint = None
            else:
                return _call_openai_compatible_with_contents(
                    contents=contents,
                    endpoint=endpoint,
                    result_parsing_func=result_parsing_func,
                    generation_configs=generation_configs,
                    check_parsed_response_not_none=check_parsed_response_not_none,
                )

    active_model_name = endpoint.model_name if endpoint is not None else model_name
    genai_client = get_genai_client(endpoint)
    for attempt in range(max_retries):
        try:
            response = genai_client.models.generate_content(
                model=active_model_name,
                contents=contents,
                config=types.GenerateContentConfig(**generation_configs),
            )

            if not response or not hasattr(response, "text") or response.text is None:
                raise ValueError("Incomplete response from Gemini API or empty text")

            raw_response = response.text
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
            error_msg = str(e)
            print(
                f"Warning: Gemini API call failed on attempt {attempt + 1} of {max_retries}. Error: {error_msg}",
            )

            if (
                "maximum number of tokens allowed" in error_msg.lower()
                or "400 bad request" in error_msg.lower()
                or "code: 400" in error_msg.lower()
                or "400 invalid_argument" in error_msg.lower()
            ):
                print("Detected input token count exceed. Not retrying.")
                raise e

            if attempt + 1 == max_retries:
                print(f"Error: All {max_retries} retry attempts failed.")
                raise e

            delay = base_interval_sec * (2 ** (attempt - 1)) if attempt > 0 else 0
            print(f"Retrying in {delay} seconds...")
            if delay > 0:
                time.sleep(delay)

    print("Error: Exited retry loop unexpectedly.")
    return {"raw_response": raw_response, "parsed_response": parsed_response}


def call_gemini_with_text_prompt(
    prompt: str,
    model_name: str,
    result_parsing_func: callable = parse_gemini_json_results,
    generation_configs: dict = {},
    check_parsed_response_not_none: bool = True,
    max_retries: int = 5,
    base_interval_sec: int = 5,
    model_role: str = model_config.MAIN_ROLE,
):
    contents = [types.Part(text=prompt)]
    return call_gemini_with_contents(
        contents=contents,
        model_name=model_name,
        model_role=model_role,
        result_parsing_func=result_parsing_func,
        generation_configs=generation_configs,
        check_parsed_response_not_none=check_parsed_response_not_none,
        max_retries=max_retries,
        base_interval_sec=base_interval_sec,
    )


def call_gemini_with_images(
    prompt: str,
    images: list[str],
    model_name: str,
    result_parsing_func: callable = parse_gemini_json_results,
    generation_configs: dict = {},
    check_parsed_response_not_none: bool = True,
    max_retries: int = 5,
    base_interval_sec: int = 5,
    model_role: str = model_config.MAIN_ROLE,
):
    llm_contents = [
        types.Part.from_text(text=prompt),
    ]
    for image_path in images:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        llm_contents.append(
            types.Part.from_bytes(data=img_bytes, mime_type="image/png")
        )

    response_dict = call_gemini_with_contents(
        contents=llm_contents,
        model_name=model_name,
        model_role=model_role,
        generation_configs=generation_configs,
        result_parsing_func=result_parsing_func,
        check_parsed_response_not_none=check_parsed_response_not_none,
        max_retries=max_retries,
        base_interval_sec=base_interval_sec,
    )
    return response_dict


def generate_image_with_gemini(
    model_name: str,
    prompt: str,
    aspect_ratio: str = "16:9",
    generation_configs: dict = None,
    max_retries: int = 5,
    base_interval_sec: int = 5,
    save_path: str = None,
) -> str:
    """
    Directly calls Gemini for image generation and returns base64.

    If the ``image`` role has a third-party endpoint enabled in
    ``model_config.yaml``, the call is routed there instead.
    """
    if generation_configs is None:
        generation_configs = {}

    endpoint = model_config.get_endpoint(model_config.IMAGE_ROLE)
    if endpoint is not None and endpoint.api_style == "openai":
        from utils.openai_utils import generate_image_with_openai_compatible

        return generate_image_with_openai_compatible(
            model_name=endpoint.model_name,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            generation_configs=generation_configs,
            base_url=endpoint.base_url,
            api_key=endpoint.api_key,
            max_retries=max_retries,
            base_interval_sec=base_interval_sec,
            save_path=save_path,
        )

    active_model_name = endpoint.model_name if endpoint is not None else model_name
    genai_client = get_genai_client(endpoint)
    for attempt in range(max_retries):
        try:
            config_args = {
                "temperature": 0.7,
                "candidate_count": 1,
                "response_modalities": ["IMAGE"],
                "image_config": types.ImageConfig(
                    aspect_ratio=aspect_ratio, image_size="1k"
                ),
            }
            config_args.update(generation_configs)

            response = genai_client.models.generate_content(
                model=active_model_name,
                contents=[types.Part.from_text(text=prompt)],
                config=types.GenerateContentConfig(**config_args),
            )
            if hasattr(response, "candidates") and response.candidates:
                if (
                    len(response.candidates) > 0
                    and response.candidates[0].content
                    and response.candidates[0].content.parts
                ):
                    for part in response.candidates[0].content.parts:
                        if part.inline_data:
                            img_base64 = base64.b64encode(part.inline_data.data).decode(
                                "utf-8"
                            )

                            if save_path:
                                try:
                                    os.makedirs(
                                        os.path.dirname(save_path), exist_ok=True
                                    )
                                    with open(save_path, "wb") as fh:
                                        fh.write(base64.b64decode(img_base64))
                                    print(
                                        f"Image for prompt '{prompt}' saved to {save_path}"
                                    )
                                except Exception as e:
                                    print(
                                        f"Warning: Failed to save image to {save_path}: {e}"
                                    )

                            return img_base64
        except Exception as e:
            print(f"Warning: Gemini API image generation failed {e}.")
            if attempt + 1 == max_retries:
                print(
                    f"Error: All {max_retries} retry attempts failed for image generation."
                )
                break

            delay = base_interval_sec * (2 ** (attempt - 1)) if attempt > 0 else 0
            print(f"Retrying in {delay} seconds...")
            if delay > 0:
                time.sleep(delay)

    return None
