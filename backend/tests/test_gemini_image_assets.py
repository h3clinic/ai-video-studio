import base64
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from real_video.gemini_image_assets import decode_image, generate_reference, MODEL, ImageHTTPError


def fixture():
    stream = io.BytesIO()
    Image.new("RGB", (256,256), "red").save(stream, format="PNG")
    return {"status":"completed", "steps":[{"type":"model_output", "content":[
        {"type":"image", "mime_type":"image/png", "data":base64.b64encode(stream.getvalue()).decode()}]}]}


class ImageAssetsTests(unittest.TestCase):
    def test_generate_content_explicit_route(self):
        block=fixture()["steps"][0]["content"][0]
        response={"candidates":[{"finishReason":"STOP","content":{"parts":[
            {"inlineData":{"mimeType":block["mime_type"],"data":block["data"]}}]}}],
            "usageMetadata":{"totalTokenCount":123}}
        calls=[]
        def transport(payload): calls.append(payload); return response
        with tempfile.TemporaryDirectory() as temp:
            meta=generate_reference(Path(temp)/"run",transport=transport,api="generate-content")
        self.assertEqual(calls[0]["generationConfig"]["maxOutputTokens"],8192)
        self.assertEqual(meta["api"],"generate-content")
        self.assertEqual(meta["usage"]["totalTokenCount"],123)
        response["candidates"][0]["finishReason"]="MAX_TOKENS"
        with self.assertRaises(ValueError): decode_image(response,"generate-content")

    def test_safe_http_status_persisted(self):
        def fail(payload): raise ImageHTTPError(403)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/"run"
            with self.assertRaises(ImageHTTPError): generate_reference(root,transport=fail)
            self.assertEqual(json.loads((root/"metadata.json").read_text())["http_status"],403)

    def test_validated_bytes(self):
        data, meta = decode_image(fixture())
        self.assertTrue(data.startswith(b'\x89PNG'))
        self.assertEqual(meta["width"],256)
        self.assertEqual(len(meta["sha256"]),64)

    def test_missing_multiple_incomplete(self):
        for bad in ({}, {"status":"in_progress"}, {"steps":fixture()["steps"]*2}):
            with self.assertRaises(ValueError): decode_image(bad)

    def test_mime_and_corrupt(self):
        bad = fixture()
        bad["steps"][0]["content"][0]["mime_type"] = "image/jpeg"
        with self.assertRaises(ValueError): decode_image(bad)
        bad = fixture()
        bad["steps"][0]["content"][0]["data"] = "bad!"
        with self.assertRaises(ValueError): decode_image(bad)

    def test_fresh_only_one_call(self):
        calls=[]
        def transport(payload): calls.append(copy.deepcopy(payload)); return fixture()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/"run"
            meta=generate_reference(root, transport=transport)
            self.assertEqual(len(calls),1)
            self.assertFalse(calls[0]["store"])
            self.assertEqual(calls[0]["model"],MODEL)
            self.assertFalse(meta["gaussian_state_modified"])
            self.assertTrue((root/meta["image"]).exists())
            with self.assertRaises(FileExistsError): generate_reference(root,transport=transport)
            self.assertEqual(len(calls),1)

    def test_failure_preserved_no_retry_no_error_leak(self):
        calls=[]
        def fail(payload): calls.append(1); raise RuntimeError("secret-must-not-be-recorded")
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/"run"
            with self.assertRaises(RuntimeError): generate_reference(root,transport=fail)
            text=(root/"metadata.json").read_text()
            self.assertNotIn("secret-must-not-be-recorded",text)
            self.assertEqual(json.loads(text)["status"],"failed")
            self.assertEqual(len(calls),1)


if __name__ == "__main__": unittest.main()
