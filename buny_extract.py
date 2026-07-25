#!/usr/bin/env python3
"""
buny_extract.py — pure-Python extractor for Star Wars: Bounty Hunter (Aspyr PC)
                  .buny archives. No Java, no QuickBMS.

Usage:
    python3 buny_extract.py <data.buny> <output-dir>
    python3 buny_extract.py <data.buny> --list
    python3 buny_extract.py <data.buny> <out> --filter .lvr .aset .mat

Needs a zstd binding. It will use, in order of preference:
    - compression.zstd   (Python 3.14+ stdlib)
    - zstandard          (pip install zstandard)
    - pyzstd             (pip install pyzstd)

FORMAT (confirmed against the ResHax BMS script AND BunyUtil's own header dump;
two independent implementations agree, so this is measured, not guessed):

  0x00  char[16]  "BunyArchTheForge"
  0x10  u64       unknown
  0x18  u64       unknown
  0x20  u64       toc_offset        (== 0x50 in the shipped archives)
  0x28  u64       toc_size          (entry count = toc_size / 40)
  0x30  u64       names_offset
  0x38  u64       names_size
  0x40  u64       toc2_offset       (purpose unknown - not needed to extract)
  0x48  u64       toc2_size

  TOC entry, 40 (0x28) bytes each:
    +0x00  u64  type          (meaning unknown)
    +0x08  u64  size          uncompressed size
    +0x10  u32  name_offset   relative to names_offset
    +0x14  u32  name_size
    +0x18  u64  offset        absolute, into the archive
    +0x20  u64  zsize         stored size (== size when the entry is not compressed)

  A compressed entry begins with a 0x18-byte header:
    +0x00  byte[16]  unknown
    +0x10  u32       chunk_count
    +0x14  u32       unknown
  followed by chunk_count * 8 bytes of chunk table, then the zstd payload.
  (Best guess: the table is a seek index. We don't need it - we just skip it.)
"""

import argparse
import os
import struct
import sys

MAGIC = b"BunyArchTheForge"
HEADER_LEN = 0x50
ENTRY_LEN = 0x28
COMP_HDR_LEN = 0x18


# ---------------------------------------------------------------------------
# zstd backend
# ---------------------------------------------------------------------------
def get_zstd_decompress():
    """Return decompress(data, expected_size) -> bytes, handling concatenated frames."""
    try:
        from compression import zstd as _z  # Python 3.14+

        def dec(data, expected):
            out = bytearray()
            pos = 0
            while pos < len(data) and len(out) < expected:
                d = _z.ZstdDecompressor()
                out += d.decompress(data[pos:])
                if d.unused_data:
                    pos = len(data) - len(d.unused_data)
                else:
                    break
            return bytes(out)

        return dec, "compression.zstd (stdlib)"
    except ImportError:
        pass

    try:
        import zstandard as _z

        def dec(data, expected):
            out = bytearray()
            pos = 0
            while pos < len(data) and len(out) < expected:
                d = _z.ZstdDecompressor().decompressobj()
                out += d.decompress(data[pos:])
                unused = getattr(d, "unused_data", b"")
                if unused:
                    pos = len(data) - len(unused)
                else:
                    break
            return bytes(out)

        return dec, "zstandard"
    except ImportError:
        pass

    try:
        import pyzstd as _z

        def dec(data, expected):
            return _z.decompress(data)

        return dec, "pyzstd"
    except ImportError:
        pass

    sys.exit(
        "[ERROR] No zstd library found.\n"
        "        Run:  py -3 -m pip install zstandard\n"
        "        (or upgrade to Python 3.14+, which has it built in)"
    )


# ---------------------------------------------------------------------------
def read_toc(f):
    head = f.read(HEADER_LEN)
    if len(head) < HEADER_LEN or head[:16] != MAGIC:
        sys.exit("[ERROR] not a .buny archive (bad magic)")

    (_u1, _u2, toc_off, toc_size,
     names_off, names_size, toc2_off, toc2_size) = struct.unpack_from("<8Q", head, 0x10)

    if toc_size % ENTRY_LEN != 0:
        sys.exit(f"[ERROR] toc_size {toc_size} is not a multiple of {ENTRY_LEN} "
                 f"- header misread, stopping rather than guessing")
    count = toc_size // ENTRY_LEN

    print(f"  toc     @ {toc_off:>12}  {toc_size:>10} bytes  -> {count} entries")
    print(f"  names   @ {names_off:>12}  {names_size:>10} bytes")
    print(f"  toc2    @ {toc2_off:>12}  {toc2_size:>10} bytes  (unused)")

    f.seek(names_off)
    names = f.read(names_size)

    f.seek(toc_off)
    raw = f.read(toc_size)

    entries = []
    for i in range(count):
        ftype, size, name_off, name_size, off, zsize = struct.unpack_from(
            "<QQIIQQ", raw, i * ENTRY_LEN)
        name = names[name_off:name_off + name_size].decode("utf-8", errors="replace")
        entries.append(dict(name=name, type=ftype, size=size, off=off, zsize=zsize))
    return entries


def extract_one(f, e, dec):
    """Return the file's bytes, or raise ValueError on a size mismatch."""
    size, off, zsize = e["size"], e["off"], e["zsize"]

    if zsize == size:
        f.seek(off)
        return f.read(size)

    f.seek(off)
    hdr = f.read(COMP_HDR_LEN)
    chunk_count = struct.unpack_from("<I", hdr, 0x10)[0]
    table_len = chunk_count * 8

    payload_off = off + COMP_HDR_LEN + table_len
    payload_len = zsize - COMP_HDR_LEN - table_len
    if payload_len < 0:
        raise ValueError(f"negative payload length ({payload_len}) - header misread")

    f.seek(payload_off)
    payload = f.read(payload_len)

    if payload_len == size:          # header present but content stored raw
        return payload

    data = dec(payload, size)
    if len(data) != size:
        raise ValueError(f"decompressed {len(data)} bytes, TOC says {size}")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("outdir", nargs="?")
    ap.add_argument("--list", action="store_true", help="list contents, extract nothing")
    ap.add_argument("--filter", nargs="+", metavar="EXT",
                    help="only extract these extensions, e.g. --filter .lvr .aset")
    args = ap.parse_args()

    if not args.list and not args.outdir:
        ap.error("need an output directory (or --list)")

    dec, backend = get_zstd_decompress()
    print(f"[ok] zstd backend: {backend}")
    print(f"[..] reading {args.archive}")

    with open(args.archive, "rb") as f:
        entries = read_toc(f)

        if args.list:
            for e in entries:
                flag = "raw " if e["zsize"] == e["size"] else "zstd"
                print(f"  {flag} {e['size']:>10}  {e['name']}")
            print(f"\n{len(entries)} entries")
            return

        wanted = entries
        if args.filter:
            exts = {x.lower() if x.startswith(".") else "." + x.lower() for x in args.filter}
            wanted = [e for e in entries
                      if os.path.splitext(e["name"])[1].lower() in exts]
            print(f"[ok] filter matched {len(wanted)} of {len(entries)} entries")

        ok = 0
        bad = []
        for i, e in enumerate(wanted, 1):
            try:
                data = extract_one(f, e, dec)
            except Exception as exc:
                bad.append((e["name"], str(exc)))
                continue

            dest = os.path.join(args.outdir, *e["name"].replace("\\", "/").split("/"))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as g:
                g.write(data)
            ok += 1

            if i % 500 == 0 or i == len(wanted):
                print(f"  [{i}/{len(wanted)}] {ok} ok, {len(bad)} failed", end="\r")

        print()
        print(f"[done] extracted {ok}/{len(wanted)}  -> {args.outdir}")

        # The acceptance test: every entry must come out at exactly its declared size.
        if bad:
            print(f"\n[!!] {len(bad)} entries did NOT match their declared size.")
            print("     This means a field is being misread. Do not ignore it.")
            for name, why in bad[:20]:
                print(f"     {name}: {why}")
            if len(bad) > 20:
                print(f"     ... and {len(bad)-20} more")
            sys.exit(2)
        else:
            print("[ok] every entry decompressed to exactly its declared size.")


if __name__ == "__main__":
    main()
