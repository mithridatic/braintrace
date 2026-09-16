"""Cell table of the H01 C3 release: every segment tagged ``neuron`` with its
layer, class, modifiers and the released count columns.

Source: the Neuroglancer segment-properties file of the C3 release
(``.../20210601/c3/segment_properties/info``, 46,637 segments, 32 tags, 12
numeric columns). The object is served gzip-compressed; the decoded JSON is
pinned by sha256 and kept under ``.cache/h01/`` (gitignored). The table is
written to ``docs/evidence/h01-c3-cell-table.json`` (compact, several MB).

Class vocabulary is the one of ``braintrace/datasets/h01_cell_types.py``
(``CLASSES`` / ``MODIFIERS`` / ``LAYERS``), loaded from the file so that the
script runs in a venv without brainstate. Class tags outside that vocabulary
(``spiny-stellate``, ``unclassified-neuron``) are kept verbatim.

Run from the worktree root on the box:
    /workspace/venv-c3/bin/python docs/evidence/h01_c3_cell_table.py
"""

import collections
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "https://storage.googleapis.com/h01-release/data/20210601/c3/segment_properties/info"
CACHE = ROOT / ".cache/h01/c3-segment-properties.json"
OUT = ROOT / "docs/evidence/h01-c3-cell-table.json"
COUNT_COLUMNS = ("NVx", "NSO", "NSI", "NSIe", "NSIi", "NDe", "NAx", "NSp")
# Class tags of the release that the runtime vocabulary does not name; kept verbatim.
EXTRA_CLASS_TAGS = ("spiny-stellate", "unclassified-neuron")
NEURON_TAG = "neuron"


def load_cell_types(root=ROOT):
    """Load ``braintrace/datasets/h01_cell_types.py`` from its file.

    The package ``__init__`` imports brainstate, which the data venv does not
    carry; the module itself needs only the standard library.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    module
    """
    path = root / "braintrace/datasets/h01_cell_types.py"
    spec = importlib.util.spec_from_file_location("h01_cell_types", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify_tags(tags, vocab):
    """Split one segment's tags into layer, class and modifiers.

    Parameters
    ----------
    tags : iterable of str
        Released tags, e.g. ``["L2", "pyramidal", "neuron", "bipolar"]``.
    vocab : module
        The ``h01_cell_types`` module (``LAYERS``, ``CLASSES``).

    Returns
    -------
    dict
        ``layer`` (one of ``LAYERS``, ``"layer-unclassified"`` or ``None``),
        ``cell_class`` (a key of ``CLASSES``, an ``EXTRA_CLASS_TAGS`` entry, or
        ``None``) and ``modifiers`` (every other tag except ``neuron``, sorted).

    Examples
    --------
    .. code-block:: python

        >>> from docs.evidence.h01_c3_cell_table import classify_tags, load_cell_types
        >>> classify_tags(["L5", "bipolar", "neuron", "pyramidal"], load_cell_types())
        {'layer': 'L5', 'cell_class': 'pyramidal', 'modifiers': ['bipolar']}
    """
    tags = list(tags)
    layers = [t for t in tags if t in vocab.LAYERS]
    if len(layers) > 1:
        raise ValueError(f"more than one layer tag: {tags}")
    layer = layers[0] if layers else ("layer-unclassified" if "layer-unclassified" in tags else None)
    classes = [t for t in tags if t in vocab.CLASSES] or [t for t in tags if t in EXTRA_CLASS_TAGS]
    if len(classes) > 1:
        raise ValueError(f"more than one class tag: {tags}")
    cell_class = classes[0] if classes else None
    used = {NEURON_TAG, "layer-unclassified", layer, cell_class}
    return {"layer": layer, "cell_class": cell_class, "modifiers": sorted(t for t in tags if t not in used)}


def decode_payload(payload):
    """Return the JSON text of a segment-properties download, gunzipping when needed."""
    if payload[:2] == b"\x1f\x8b":
        payload = gzip.decompress(payload)
    return payload.decode("utf-8")


def fetch(url=SOURCE):
    """Download the segment-properties object as bytes."""
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


def neuron_rows(info, vocab):
    """Rows for every segment tagged ``neuron``.

    Parameters
    ----------
    info : dict
        Parsed segment-properties JSON (``inline.ids``, ``inline.properties``).
    vocab : module
        The ``h01_cell_types`` module.

    Returns
    -------
    list of dict
        ``id`` (int), ``layer``, ``cell_class``, ``modifiers`` and the
        ``COUNT_COLUMNS`` values, in the file's order.
    """
    inline = info["inline"]
    props = {p["id"]: p for p in inline["properties"]}
    tag_names = props["tags"]["tags"]
    rows = []
    for k, sid in enumerate(inline["ids"]):
        tags = [tag_names[t] for t in props["tags"]["values"][k]]
        if NEURON_TAG not in tags:
            continue
        row = {"id": int(sid), **classify_tags(tags, vocab)}
        row.update({c: props[c]["values"][k] for c in COUNT_COLUMNS})
        rows.append(row)
    return rows


def build_table(text, vocab, sha256, fetched_utc, source=SOURCE):
    """Assemble the output document from the decoded JSON text."""
    rows = neuron_rows(json.loads(text), vocab)
    counts = collections.Counter(str(r["cell_class"]) for r in rows)
    return {"source": source, "sha256": sha256, "fetched_utc": fetched_utc, "neurons": len(rows),
            "counts_by_class": dict(sorted(counts.items())), "count_columns": list(COUNT_COLUMNS),
            "rows": rows}


def main():
    vocab = load_cell_types()
    fetched = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    text = decode_payload(fetch())
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(CACHE.read_bytes()).hexdigest()
    table = build_table(text, vocab, digest, fetched)
    OUT.write_text(json.dumps(table, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in table.items() if k != "rows"}, indent=1))
    print("wrote", OUT, OUT.stat().st_size, "bytes; cache", CACHE, "sha256", digest)


if __name__ == "__main__":
    sys.exit(main())
