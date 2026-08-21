"""Immutability checks for historical Example 21 evidence bundles."""

from __future__ import annotations

import hashlib
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[2]
HISTORICAL = {
    "example21-default-20260821": {
        "data_manifest.json": "fb3aae51ce535213a2ade3fdba75cbfc6f5c2f4d168a306d07e0a72d52b7446c",
        "latent_reasoning.png": "43a39e1136350ab2a7c2233a3a4c2aa6d2fb0e19dae85e9fa761472cc52c087d",
        "report.txt": "3a54b1c4ac538432428ef7f304c7caa73e6c2dd17f44e41c14d311edcd50a28d",
        "result.json": "dbce78a9b50a663995da7b458dfa8cd05dee93e5b41d823933a4b79c32f34a78",
    },
    "example21-ei-dale-010-20260821": {
        "data_manifest.json": "fb3aae51ce535213a2ade3fdba75cbfc6f5c2f4d168a306d07e0a72d52b7446c",
        "latent_reasoning.png": "36b354636a66fa1fa6225704861a366c92edf71713c0c0b7ebdb4c9232a451cd",
        "report.txt": "248b27932f064812d6dc59607a738b7f3097daf729517a9e754c84b9342d7fb6",
        "result.json": "8c1e9da06a50355c4828d562e0054f52f7d197f99a8b0cadde634a09f4a4da28",
    },
    "example21-ei-dale-020-20260821": {
        "data_manifest.json": "fb3aae51ce535213a2ade3fdba75cbfc6f5c2f4d168a306d07e0a72d52b7446c",
        "latent_reasoning.png": "437f5d040faf3e057206165aabb032275aabc07929251708ee6b5fccb927b1cf",
        "report.txt": "e1c9ef11a705094401ab5a7a90fc41af5b7f7c865ae1fcd86e81a97227cf9555",
        "result.json": "ef5c2e93b8373678b0b957a3e47220f538ab55874e6c9a32726b091db9cd1b68",
    },
    "example21-ei-dale-070-20260821": {
        "data_manifest.json": "fb3aae51ce535213a2ade3fdba75cbfc6f5c2f4d168a306d07e0a72d52b7446c",
        "latent_reasoning.png": "a151f1c1db86913b17740c956c2b86e80ad622e9f632fbe7ca5dafa99fa92691",
        "report.txt": "055dcf80ce7bf9c94809b255631b421deaabf9027e708a6a02ba6a0c5190b48f",
        "result.json": "f57637b13fbe8ad6bd69171c9d48dd3b8fca7833a71f327e865cdb73a8909e40",
    },
    "example21-ei-dale-090-20260821": {
        "data_manifest.json": "fb3aae51ce535213a2ade3fdba75cbfc6f5c2f4d168a306d07e0a72d52b7446c",
        "latent_reasoning.png": "0841854c8c80be0e44cf78dee592778211d5a930b12109ae61a4294d8f427cbc",
        "report.txt": "fe4349343d38dbbfe16f6067a277c9cbae9f0836639ae698b34eaf53f0f878e1",
        "result.json": "6c3e38a2dfdb5b06cd06ad5e9949620084e03bef624910bc403098881c9fd959",
    },
    "example21-ei-dale-20260821": {
        "data_manifest.json": "fb3aae51ce535213a2ade3fdba75cbfc6f5c2f4d168a306d07e0a72d52b7446c",
        "latent_reasoning.png": "02a0d41a5f5f346a455d06cf9a43cbdac73de74231418636b1d3120d364ea1ab",
        "report.txt": "826f4b42917634da1b52debdf05f98952001680ea1912406278f65a7ada70630",
        "result.json": "2a24bed3518d93593ecf77fffa027c2f65512a7688596c85ab64ee74c16437ab",
    },
}


def test_historical_schema1_evidence_is_byte_identical() -> None:
    for directory, files in HISTORICAL.items():
        path = ROOT / "var" / directory
        assert {item.name for item in path.iterdir() if item.is_file()} == set(files)
        for name, expected in files.items():
            assert hashlib.sha256((path / name).read_bytes()).hexdigest() == expected
