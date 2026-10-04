"""Pure validation tests for render-demo.py; no media, network, or FFmpeg."""

import copy
import importlib.util
from pathlib import Path
import unittest


RENDERER_PATH = Path(__file__).resolve().parents[1] / "render-demo.py"
SPEC = importlib.util.spec_from_file_location("render_demo", RENDERER_PATH)
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def outline_for(boundaries=(0, 30, 60, 95, 130, 160, 180)):
    total = len(boundaries) - 1
    segments = [
        {
            "index": index,
            "start_seconds": start,
            "end_seconds": end,
            "title": f"Section {index}",
            "narration": "A truthful demonstration.",
            "total_segments": total,
        }
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1)
    ]
    return {"segments": segments, "spoken_word_count": 3 * total}


def timed_script(boundaries=(0, 30, 60, 95, 130, 160, 180)):
    def timestamp(value):
        return f"{value // 60:02d}:{value % 60:02d}"

    return "Narration guide; headings below define the timeline.\n\n" + "\n\n".join(
        f"{timestamp(start)}–{timestamp(end)} — Section {index}\n\n"
        "A truthful\ndemonstration."
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1)
    )


class TimedScriptTests(unittest.TestCase):
    def test_six_sections_have_exact_timing_normalized_narration_and_word_count(self):
        self.assertEqual(renderer.parse_timed_script(timed_script()), outline_for())

    def test_crlf_input_preserves_timing(self):
        self.assertEqual(
            renderer.parse_timed_script(timed_script().replace("\n", "\r\n")),
            outline_for(),
        )

    def test_absent_timed_headings_are_rejected(self):
        for source in ("", "Untimed narration only."):
            with self.subTest(source=source), self.assertRaises(ValueError):
                renderer.parse_timed_script(source)

    def test_seconds_component_cannot_overflow_into_next_minute(self):
        source = timed_script()
        for invalid in (
            source.replace("01:00", "00:60"),
            source.replace("01:35", "00:95"),
        ):
            with self.subTest(source=invalid), self.assertRaises(ValueError):
                renderer.parse_timed_script(invalid)

    def test_missing_narration_is_rejected(self):
        for source in (
            "00:00–03:00 — Silent section\n\n",
            timed_script().replace("A truthful\ndemonstration.", "", 1),
        ):
            with self.subTest(source=source), self.assertRaises(ValueError):
                renderer.parse_timed_script(source)

    def test_gaps_overlaps_and_nonpositive_sections_are_rejected(self):
        source = timed_script()
        malformed = (
            source.replace("00:00–00:30", "00:01–00:30"),
            source.replace("00:30–01:00", "00:31–01:00"),
            source.replace("00:30–01:00", "00:29–01:00"),
            source.replace("00:30–01:00", "00:30–00:30"),
            source.replace("01:00–01:35", "01:35–01:00"),
            source.replace("02:40–03:00", "02:40–02:59"),
            source.replace("02:40–03:00", "02:40–03:01"),
        )
        for invalid in malformed:
            with self.subTest(source=invalid), self.assertRaises(ValueError):
                renderer.parse_timed_script(invalid)


class OutlineTests(unittest.TestCase):
    def test_valid_outline_needs_no_media(self):
        renderer.validate_outline(outline_for())
        renderer.validate_outline(outline_for((0, 180)))

    def test_empty_or_missing_segments_are_rejected(self):
        for outline in ({}, {"segments": [], "spoken_word_count": 0}):
            with self.subTest(outline=outline), self.assertRaises(ValueError):
                renderer.validate_outline(outline)

    def test_invalid_timeline_is_rejected(self):
        cases = (
            (0, "start_seconds", 1),
            (1, "start_seconds", 29),
            (1, "start_seconds", 31),
            (1, "end_seconds", 30),
            (1, "end_seconds", 20),
            (5, "end_seconds", 179),
            (5, "end_seconds", 181),
        )
        for index, key, value in cases:
            outline = outline_for()
            outline["segments"][index][key] = value
            with self.subTest(index=index, key=key, value=value), self.assertRaises(ValueError):
                renderer.validate_outline(outline)

    def test_missing_or_blank_narration_is_rejected(self):
        for narration in (None, "", " \n\t "):
            outline = outline_for()
            if narration is None:
                del outline["segments"][2]["narration"]
            else:
                outline["segments"][2]["narration"] = narration
            with self.subTest(narration=narration), self.assertRaises(ValueError):
                renderer.validate_outline(outline)

    def test_incorrect_spoken_word_count_is_rejected(self):
        outline = outline_for()
        outline["spoken_word_count"] += 1
        with self.assertRaises(ValueError):
            renderer.validate_outline(outline)


class ShotTests(unittest.TestCase):
    def assert_complete_coverage(self, shots):
        self.assertTrue(shots)
        self.assertEqual(shots[0]["start"], 0)
        self.assertEqual(shots[-1]["end"], 180)
        self.assertEqual(sum(shot["end"] - shot["start"] for shot in shots), 180)
        for shot in shots:
            self.assertLess(shot["start"], shot["end"])
            self.assertGreaterEqual(shot["start"], shot["segment"]["start_seconds"])
            self.assertLessEqual(shot["end"], shot["segment"]["end_seconds"])
        for previous, following in zip(shots, shots[1:]):
            self.assertEqual(previous["end"], following["start"])

    def test_current_six_section_layout_covers_180_seconds_in_13_shots(self):
        outline = outline_for()
        original = copy.deepcopy(outline)
        shots = renderer.build_shots(outline)
        self.assertEqual(len(shots), 13)
        self.assert_complete_coverage(shots)
        self.assertEqual(outline, original)

    def test_nine_section_layout_also_covers_180_seconds(self):
        self.assert_complete_coverage(renderer.build_shots(outline_for(tuple(range(0, 181, 20)))))

    def test_incompatible_short_segments_are_rejected(self):
        # Each outline itself is contiguous and positive; only its editorial
        # slot lengths are incompatible with the renderer's fixed cut points.
        for boundaries in (
            (0, 5, 60, 95, 130, 160, 180),
            (0, 30, 35, 95, 130, 160, 180),
            (0, 30, 60, 89, 130, 160, 180),
            (0, 30, 60, 95, 110, 160, 180),
            (0, 30, 60, 95, 130, 145, 180),
        ):
            outline = outline_for(boundaries)
            renderer.validate_outline(outline)
            with self.subTest(boundaries=boundaries), self.assertRaises(ValueError):
                renderer.build_shots(outline)


class CropTests(unittest.TestCase):
    capture_info = {"width": 1920, "height": 1080, "seconds": 180, "audio": False}

    def test_omitted_crop_and_valid_edges(self):
        self.assertIsNone(renderer.validate_crop(None, self.capture_info))
        for crop in ([0, 0, 1920, 1080], [1918, 1078, 2, 2], (10, 20, 100, 200)):
            with self.subTest(crop=crop):
                self.assertEqual(renderer.validate_crop(crop, self.capture_info), tuple(crop))

    def test_crop_requires_four_integer_values_not_booleans(self):
        invalid = ([], [0, 0, 100], [0, 0, 100, 100, 0], "0,0,100,100", 0,
                   [False, 0, 100, 100], [0, True, 100, 100],
                   [0, 0, True, 100], [0, 0, 100, False],
                   [0.0, 0, 100, 100], [0, 0, "100", 100], [0, None, 100, 100])
        for crop in invalid:
            with self.subTest(crop=crop), self.assertRaises(ValueError):
                renderer.validate_crop(crop, self.capture_info)

    def test_crop_rejects_negative_small_and_out_of_bounds_rectangles(self):
        invalid = ([-1, 0, 100, 100], [0, -1, 100, 100], [0, 0, -2, 100],
                   [0, 0, 100, -2], [0, 0, 1, 100], [0, 0, 100, 1],
                   [0, 0, 1921, 1080], [0, 0, 1920, 1081],
                   [1919, 0, 2, 100], [0, 1079, 100, 2])
        for crop in invalid:
            with self.subTest(crop=crop), self.assertRaises(ValueError):
                renderer.validate_crop(crop, self.capture_info)


class GithubUrlTests(unittest.TestCase):
    def test_valid_repository_and_branch_urls_are_returned_unchanged(self):
        for value in ("", "https://github.com/owner/repo", "https://github.com/owner/repo/",
                      "https://github.com/h3clinic/ai-video-studio",
                      "https://github.com/owner/repo/tree/main",
                      "https://github.com/owner/repo/tree/codex/sponsor-demo-20261004"):
            with self.subTest(value=value):
                self.assertEqual(renderer.validate_github_url(value), value)

    def test_non_github_or_non_https_urls_are_rejected(self):
        for value in ("http://github.com/owner/repo", "https://example.com/owner/repo",
                      "https://github.com.evil.example/owner/repo", "//github.com/owner/repo",
                      "file:///owner/repo", "github.com/owner/repo"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                renderer.validate_github_url(value)

    def test_credentials_ports_queries_and_fragments_are_rejected(self):
        for value in ("https://user:secret@github.com/owner/repo",
                      "https://user@github.com/owner/repo", "https://github.com:443/owner/repo",
                      "https://github.com/owner/repo?tab=readme", "https://github.com/owner/repo#readme"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                renderer.validate_github_url(value)

    def test_invalid_repository_shapes_are_rejected(self):
        for path in ("", "/owner", "/owner/", "/owner//repo", "/owner/repo/extra",
                     "/owner/repo/tree", "/owner/repo/tree/", "/owner/repo/blob/main",
                     "/owner/bad repo", "/owner/bad?repo", "/owner/.", "/owner/.."):
            with self.subTest(path=path), self.assertRaises(ValueError):
                renderer.validate_github_url("https://github.com" + path)

    def test_traversal_and_backslashes_are_rejected(self):
        for path in ("/owner/repo/tree/../main", "/owner/repo/tree/a/./b",
                     "/owner/repo/tree/%2e%2e/main", "/owner/repo/tree/a\\b",
                     "/owner\\name/repo", "/owner/repo/tree/a%5cb"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                renderer.validate_github_url("https://github.com" + path)


if __name__ == "__main__":
    unittest.main()
