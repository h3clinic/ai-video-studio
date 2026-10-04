"""One bounded Gemini image request for reference material, NOT a 3D asset.

REST schema checked against ai.google.dev/gemini-api/docs/image-generation
and interactions-overview on 2026-10-04. No retry, alternate model, or GPU.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from PIL import Image
from .gemini_edit_planner import NoRedirect

MODEL = "gemini-3.1-flash-image"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
MAX_RESPONSE = 24 * 2**20
PROMPT = """Create ONE photorealistic material-reference contact sheet for a red apple,
with four clearly separated equal quadrants on neutral grey and no labels/text.
Top left: whole ripe red apple, short brown stem, detailed natural skin pores,
subtle red striations and realistic waxy highlights, three-quarter view.
Top right: cut apple slices, moist ivory-white flesh and a very thin red skin edge,
NOT citrus wedges, no orange pith or membranes.
Bottom left: thin curved red apple peel ribbons with subtle ivory underside,
NOT thick orange rind. Bottom right: a small irregular bitten apple fragment
with moist pale flesh and red skin, suitable for a donkey's mouth-held bite.
Consistent neutral daylight, crisp macro surface detail, natural proportions.
No donkey, bowl, scenery, hands, extra fruit or orange material. This is only
a 2D appearance reference for later Gaussian fitting, not a 3D model or video."""


class ImageHTTPError(RuntimeError):
    def __init__(self, status):
        self.http_status = int(status)
        super().__init__(f"Gemini image HTTP {self.http_status}; no automatic retry")


def _request(payload, key_loader=None, api="interactions"):
    from .runway_settings import CredentialStore
    key = key_loader() if key_loader else CredentialStore("gemini").load_for_api()
    endpoint = ENDPOINT if api == "interactions" else "https://generativelanguage.googleapis.com/v1beta/models/"+MODEL+":generateContent"
    request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": key}, method="POST")
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect).open(request, timeout=180) as response:
            raw = response.read(MAX_RESPONSE + 1)
        if len(raw) > MAX_RESPONSE:
            raise RuntimeError("oversize")
        return json.loads(raw)
    except urllib.error.HTTPError as error:
        raise ImageHTTPError(error.code) from None
    except Exception:
        raise RuntimeError("Gemini image transport failed; no automatic retry") from None


def decode_image(result, api="interactions"):
    if api == "generate-content":
        candidates = result.get("candidates", []) if isinstance(result, dict) else []
        if len(candidates) != 1 or candidates[0].get("finishReason") != "STOP":
            raise ValueError("Incomplete or blocked generateContent image answer")
        blocks = [{"type":"image", "mime_type":p["inlineData"].get("mimeType"),
                   "data":p["inlineData"].get("data")} for p in candidates[0].get("content",{}).get("parts",[])
                  if "inlineData" in p and not p.get("thought")]
        result = {"status":"completed", "steps":[{"type":"model_output", "content":blocks}]}
    if not isinstance(result, dict) or result.get("status", "completed") != "completed":
        raise ValueError("Incomplete Gemini image interaction")
    images = [block for step in result.get("steps", []) if step.get("type") == "model_output"
              for block in step.get("content", []) if block.get("type") == "image"]
    if len(images) != 1:
        raise ValueError("Expected exactly one generated image in model_output steps")
    block = images[0]
    mime = block.get("mime_type")
    if mime not in ("image/png", "image/jpeg"):
        raise ValueError("PNG or JPEG image required")
    encoded = block.get("data")
    if not isinstance(encoded, str) or len(encoded) > MAX_RESPONSE:
        raise ValueError("Invalid or oversized image data")
    try:
        data = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(data)) as im:
            width, height = im.size
            if not 256 <= width <= 4096 or not 256 <= height <= 4096 or width*height > 16_777_216:
                raise ValueError("Image dimensions outside bounded range")
            if im.format != {"image/png":"PNG", "image/jpeg":"JPEG"}[mime]:
                raise ValueError("Declared image MIME does not match bytes")
            im.verify()
    except Exception:
        raise ValueError("Invalid image bytes, dimensions or MIME") from None
    return data, {"mime_type": mime, "width": width, "height": height,
                  "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def generate_reference(output_dir, *, transport=None, key_loader=None, api="interactions", prompt=PROMPT):
    """Create a fresh run directory before its only request; never repeat it.

    transport(payload) enables offline tests. Only controlled request metadata
    and image bytes are persisted, never the raw provider response or key.
    """
    if api not in ("interactions", "generate-content"):
        raise ValueError("Explicit supported API route required")
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 4000:
        raise ValueError("Bounded image prompt required")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    payload = {"model": MODEL, "input": [{"type":"text", "text":prompt}],
               "store": False, "response_format": {"type":"image", "aspect_ratio":"1:1", "image_size":"1K"}}
    if api == "generate-content":
        payload = {"contents":[{"role":"user", "parts":[{"text":prompt}]}],
                   "generationConfig":{"maxOutputTokens":8192,"responseModalities":["IMAGE"],
                                       "imageConfig":{"aspectRatio":"1:1","imageSize":"1K"}}}
    metadata = {"model": MODEL, "prompt": prompt, "request_count":1,
                "asset_type":"2D material reference only", "gaussian_state_modified":False,
                "geometry_verified":False, "visual_review":"pending", "status":"started", "api":api,
                "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest()}
    record = root/"metadata.json"
    record.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    started = time.perf_counter()
    try:
        result = transport(payload) if transport else _request(payload, key_loader, api)
        data, image_meta = decode_image(result, api)
        name = "apple_parts_reference" + (".png" if image_meta["mime_type"] == "image/png" else ".jpg")
        with (root/name).open("xb") as image_file:
            image_file.write(data)
        usage = result.get("usageMetadata" if api == "generate-content" else "usage", {})
        metadata.update(status="generated_unreviewed", image=name, **image_meta,
                        usage={k:v for k,v in usage.items() if type(v) in (int,float)} if isinstance(usage,dict) else {})
    except Exception as error:
        metadata.update(status="failed", error_type=type(error).__name__)
        if isinstance(error, ImageHTTPError):
            metadata["http_status"] = error.http_status
        raise
    finally:
        metadata["seconds"] = time.perf_counter()-started
        record.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--live", action="store_true", help="Explicitly permit the single Gemini image API call")
    parser.add_argument("--api", choices=("interactions", "generate-content"), default="interactions")
    args = parser.parse_args()
    if not args.live:
        parser.error("--live is required; no API call made")
    try:
        print(json.dumps(generate_reference(args.output_dir, api=args.api), indent=2))
    except Exception as error:
        print(json.dumps({"status":"failed", "error_type":type(error).__name__, "retry":False}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
