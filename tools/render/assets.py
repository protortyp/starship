"""Download the CC0 Poly Haven assets the render scene uses (sky HDRI, concrete, coastal ground).

Runs inside Blender's Python or standalone:

    python tools/render/assets.py [--dir out/render/assets] [--hdri-res 4k] [--tex-res 2k]

Files land in `<dir>/<asset>/<map>.<ext>` and are reused when present. Every asset on
https://polyhaven.com is CC0 1.0 (public domain); see tools/render/README.md.
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

API = "https://api.polyhaven.com/files/"
USER_AGENT = "starship-render/0.1 (+https://github.com/)"

# asset id -> maps to fetch (Poly Haven map names) and the resolution key family.
ASSETS = {
    "qwantani_dusk_2_puresky": {"kind": "hdri", "maps": ["hdr"]},
    "belfast_sunset_puresky": {"kind": "hdri", "maps": ["hdr"]},
    "concrete_floor_worn_001": {
        "kind": "texture",
        "maps": ["Diffuse", "Rough", "nor_gl", "Displacement"],
    },
    "coast_sand_rocks_02": {
        "kind": "texture",
        "maps": ["Diffuse", "Rough", "nor_gl", "Displacement"],
    },
}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _file_entry(files: dict, asset: dict, name: str, res: str) -> tuple[str, str] | None:
    """(url, extension) of one map at a resolution, falling back to a lower one if needed."""
    if asset["kind"] == "hdri":
        table = files.get("hdri", {})
        for r in _fallback(res):
            entry = table.get(r, {}).get(name)
            if entry:
                return entry["url"], name
        return None
    table = files.get(name, {})
    for r in _fallback(res):
        entry = table.get(r)
        if entry:
            fmt = "jpg" if "jpg" in entry else next(iter(entry))
            return entry[fmt]["url"], fmt
    return None


def _fallback(res: str) -> list[str]:
    order = ["24k", "16k", "8k", "4k", "2k", "1k"]
    return order[order.index(res) :] if res in order else [res, "2k", "1k"]


def fetch_assets(
    directory: str | Path,
    hdri_res: str = "4k",
    tex_res: str = "2k",
    quiet: bool = False,
    hdri: str | None = None,
) -> dict[str, dict[str, Path]]:
    """Fetch the textures and one sky (`hdri`, default: every sky listed in ASSETS)."""
    directory = Path(directory)
    out: dict[str, dict[str, Path]] = {}
    wanted = dict(ASSETS)
    if hdri is not None:
        wanted = {k: v for k, v in wanted.items() if v["kind"] != "hdri"}
        wanted[hdri] = {"kind": "hdri", "maps": ["hdr"]}
    for asset_id, spec in wanted.items():
        folder = directory / asset_id
        folder.mkdir(parents=True, exist_ok=True)
        res = hdri_res if spec["kind"] == "hdri" else tex_res
        paths: dict[str, Path] = {}
        files = None
        for name in spec["maps"]:
            existing = list(folder.glob(f"{name}.*"))
            if existing:
                paths[name] = existing[0]
                continue
            if files is None:
                files = json.loads(_get(API + asset_id))
            entry = _file_entry(files, spec, name, res)
            if entry is None:
                if not quiet:
                    print(f"  {asset_id}: no {name} map, skipped", file=sys.stderr)
                continue
            url, ext = entry
            target = folder / f"{name}.{ext}"
            if not quiet:
                print(f"  downloading {asset_id}/{name}.{ext} ({res})")
            target.write_bytes(_get(url))
            paths[name] = target
        (folder / "LICENSE.txt").write_text(
            f"{asset_id} from https://polyhaven.com/a/{asset_id}\nLicense: CC0 1.0 Universal\n"
        )
        out[asset_id] = paths
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="out/render/assets")
    parser.add_argument("--hdri-res", default="4k")
    parser.add_argument("--tex-res", default="2k")
    args = parser.parse_args()
    for asset, maps in fetch_assets(args.dir, args.hdri_res, args.tex_res).items():
        print(asset, {k: str(v) for k, v in maps.items()})


if __name__ == "__main__":
    main()
