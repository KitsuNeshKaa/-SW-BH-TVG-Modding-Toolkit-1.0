#!/usr/bin/env python3
r"""
scan_materials.py — audit every .mat file in the unpacked SWBH tree.

Walks the whole tree once, parses every .mat the same way the Blender addon
does (key: value lines), and reports:
  - every distinct ShaderType found, with a count and a few example files
  - every OTHER key seen across all .mat files (so we notice anything the
    addon doesn't know about yet - not just ShaderType)
  - for each ShaderType, whether DiffuseMap values carry a file extension or
    not (both forms have shown up already)

Run it once against the top of the unpacked tree and send back the report
file (or paste the console output) - that's enough to extend the addon's
shader-type handling in a single pass instead of chasing one .mat at a time.

Usage:
    python3 scan_materials.py "D:\...\STAR WARS Bounty Hunter\_unpacked"
    python3 scan_materials.py "D:\...\_unpacked" --out materials_report.txt

Pure standard library, no dependencies. Safe to run outside Blender.
"""

import argparse
import os
import re
import sys
from collections import Counter, defaultdict

KV_RE = re.compile(r"\s*(\w+)\s*:\s*(.+?)\s*$")


def parse_mat(path):
    """Same parsing the addon's parse_mat() does: lowercase keys, raw values."""
    out = {}
    try:
        with open(path, "r", encoding="ascii", errors="replace") as f:
            for line in f:
                m = KV_RE.match(line)
                if m:
                    out[m.group(1).lower()] = m.group(2).strip()
    except OSError as exc:
        print("[!!] could not read %s: %s" % (path, exc), file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="top of the unpacked tree to scan")
    ap.add_argument("--out", default="materials_report.txt",
                    help="report file to write (default: materials_report.txt)")
    ap.add_argument("--examples", type=int, default=5,
                    help="example file paths to keep per shader type (default: 5)")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit("[ERROR] not a directory: %s" % args.root)

    by_shader = defaultdict(list)     # shadertype (raw casing) -> [path, ...]
    other_keys = Counter()            # key name -> count, excluding shadertype/diffusemap
    diffusemap_ext_by_shader = defaultdict(Counter)   # shadertype -> {"has_ext": n, "no_ext": n}
    all_keys_seen = Counter()
    total_files = 0
    parse_failures = []
    no_shadertype = []

    print("[..] scanning %s" % args.root)
    for dirpath, _dn, filenames in os.walk(args.root):
        for fn in filenames:
            if not fn.lower().endswith(".mat"):
                continue
            total_files += 1
            path = os.path.join(dirpath, fn)
            info = parse_mat(path)
            if not info:
                parse_failures.append(path)
                continue

            for k in info:
                all_keys_seen[k] += 1
                if k not in ("shadertype", "diffusemap"):
                    other_keys[k] += 1

            st = info.get("shadertype")
            if not st:
                no_shadertype.append(path)
                continue

            by_shader[st].append(path)

            dm = info.get("diffusemap", "")
            has_ext = bool(os.path.splitext(dm)[1])
            diffusemap_ext_by_shader[st]["has_ext" if has_ext else "no_ext"] += 1

    lines = []
    def out(s=""):
        lines.append(s)
        print(s)

    out("=" * 78)
    out("SWBH .mat AUDIT")
    out("root: %s" % args.root)
    out("total .mat files found: %d" % total_files)
    out("parse failures (unreadable): %d" % len(parse_failures))
    out("files with no ShaderType key at all: %d" % len(no_shadertype))
    out("=" * 78)
    out()

    out("--- every key seen across all .mat files (not just ShaderType) ---")
    for k, n in all_keys_seen.most_common():
        out("  %-20s %d files" % (k, n))
    out()
    if other_keys:
        out("--- keys OTHER than ShaderType/DiffuseMap (things the addon may ignore) ---")
        for k, n in other_keys.most_common():
            out("  %-20s %d files" % (k, n))
        out()

    out("--- every distinct ShaderType (%d total) ---" % len(by_shader))
    for st, paths in sorted(by_shader.items(), key=lambda kv: -len(kv[1])):
        ext_counts = diffusemap_ext_by_shader[st]
        out("")
        out("ShaderType: %r  (%d files)" % (st, len(paths)))
        out("  DiffuseMap has extension: %d, no extension: %d"
            % (ext_counts["has_ext"], ext_counts["no_ext"]))
        out("  examples:")
        for p in paths[:args.examples]:
            info = parse_mat(p)
            extra = {k: v for k, v in info.items()
                    if k not in ("shadertype", "diffusemap")}
            out("    %s" % p)
            out("      DiffuseMap: %r  %s"
                % (info.get("diffusemap"), ("  other: %s" % extra) if extra else ""))

    if no_shadertype:
        out()
        out("--- files with NO ShaderType key (first %d) ---" % args.examples)
        for p in no_shadertype[:args.examples]:
            out("  %s  raw: %s" % (p, parse_mat(p)))

    if parse_failures:
        out()
        out("--- unreadable files (first %d) ---" % args.examples)
        for p in parse_failures[:args.examples]:
            out("  %s" % p)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print()
    print("[ok] report written to %s" % args.out)


if __name__ == "__main__":
    main()
