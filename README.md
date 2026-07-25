# Star Wars: Bounty Hunter — Blender Importer (v1.0)

A Blender add-on (`io_import_swbh.py`) that imports the proprietary mesh,
animation, and level formats used by *Star Wars: Bounty Hunter* (Aspyr PC
remaster), reverse-engineered from scratch with no prior public documentation
of these specific formats.

This is a first release. It covers everything needed to bring a full level —
geometry, lights, skydome, placed props/actors, and mechanism/character
animation — into Blender in one click, plus standalone mesh/animation import.
It also has known gaps, listed honestly below rather than hidden.

---

## 1. Prerequisites: unpacking the game files

This add-on only imports files that are **already unpacked** from the game's
archives. It does not read the archive format itself.

Use that toolkit (or an equivalent extractor for this engine's archive
format) to produce a folder tree that looks like:

```
_unpacked/
  data/
    data/
      actors/
        characters/<name>/<name>.bin, materials/, animations/
        mechanisms/<letter>/<name>/<name>.bin, materials/, animations/
        props/...
        pickups/...
      worlds/
        <world>/<area>/<area>.lvt, <area>__pa_*.bin, materials/, textures/
```

The add-on expects roughly this layout (a `.bin`'s own `materials/` and
`textures/` folders next to it, or one level up) to resolve materials and
textures. Point the add-on at the root of this tree once via its preferences
(**Data root**), and every importer below will search it automatically.

---

## 2. Installing the add-on

1. Blender **Edit > Preferences > Add-ons > Install...**, pick
   `io_import_swbh.py`.
2. Enable it, then set **Data root** in its preferences to the top of your
   `_unpacked` folder (this lets texture/material lookups work from
   anywhere, not just next to the file you're importing).

Three menu entries appear under **File > Import**:

| Entry | Use for |
|---|---|
| **SW: Bounty Hunter Mesh (.bin)** | One or several individual mesh files — a character, prop, door, or room part. |
| **SW: Bounty Hunter Level Area (.lvt)** | An entire level area in one click: scans the directory, reads the `.lvt`, places every room part, light, skydome, and actor/prop. |
| **SW: Bounty Hunter Animation (.abin)** | Apply a separate animation clip onto an already-imported armature. |

---

## 3. File formats covered

| Format | What it is | Notes |
|---|---|---|
| `.bin` (modern, `0xAAAAAAAA` header) | Actor/prop/room mesh | Serialized x64 memory image; every internal offset is relative to byte 8. Auto-detected. |
| `.bin` (legacy, `"Bin file Version 16Mar01.3"`) | Older world container | Absolute offsets, no shift. Only a handful of leftover dev files use this; `gs_world.bin` (one per world, pre-portal-split) and some orphaned `pa_area_*` files. |
| `.abin` (modern, `0xABADDEED` header) | Skeleton animation clip | Per-clip NLA strips; quantized position/rotation streams. |
| `.abin` (legacy, `"ABin file Version 18Mar00.001"`) | Older animation container | A handful of prop-interaction clips use this instead. |
| `.lvt` | Level table (self-describing ASCII) | World/sky/fog, region list, lights, actor placements, geometry manifest. This is what the whole-area importer reads. |
| `.lvr` | Subset of `.lvt` | Actor placements only; redundant wherever `.lvt` exists. |

`parse_bin()` and `parse_abin()` auto-dispatch on the file's own header, so
you never need to tell the add-on which variant a file is.

Formats seen but **not used for import** (no geometry, or not needed for a
working scene): `.ass`, `.plc` (pool-size config), `.sym` (AI/waypoint
symbol table), `.vfx` (effect name list), `.lgc` (see §5).

---

## 4. Features

- **Mesh import**: modern and legacy `.bin`, single file or batch, with
  automatic LOD detection (bit-encoded in the mesh format) and collision-hull
  detection (untextured geometry, or textures prefixed `_`, e.g.
  `_Collision.tga`).
- **Whole-area import**: one `.lvt` click imports every room part on disk
  (even ones not listed in the level's own geometry manifest — the manifest
  is informational, not authoritative), the skydome, every light, and every
  placed actor/prop at its correct position and rotation.
- **Materials**: all 16 distinct `ShaderType` values found across a full
  material audit (798 files) are handled — cutout/transparent/additive/
  unlit/skydome/reflective/default — each mapped to an appropriate Blender
  node setup (Principled BSDF or Emission, with alpha-blend/clip as needed).
- **Bump/normal maps**: resolved by filename convention
  (`<DiffuseStem>NormalMap.dds`), since no `.mat` file in the whole tree
  carries an explicit bump-map key.
- **Baked vertex-color lighting**: level geometry's per-vertex baked light
  is imported and multiplied into the material.
- **Rigid mechanism animation** (doors, gates, fans, signs — anything
  built from multiple rigid, non-skinned parts tied together by a node
  hierarchy): a real Blender armature is built and animated, supporting
  both translation-only (sliding) and rotation-only (hinge/swing/spin)
  mechanisms, from whichever `.abin` sits beside the mesh.
- **Character animation**: skeleton + one or many animation clips on the
  same timeline, with FPS/gap/start/count controls.
- **Split into connected parts**: a room `.bin` can be split so each
  material/prop becomes its own object instead of one giant mesh — useful
  for isolating a specific prop, off by default.
- A **material-audit script** (`scan_materials.py`, included) walks an
  entire unpacked tree and reports every distinct `ShaderType` in use, for
  extending shader support in one pass rather than chasing files one at a
  time.

---

## 5. Known issues and limitations

Documented honestly, not swept under the rug:

1. **The Coruscant `O_B_HorizontalDoorC` / `O_B_HorizontalDoorD` doors are
   quirky.** They show a spurious ~90° twist in their own raw mesh data
   that isn't present in structurally identical doors from the same folder
   (`O_B_HorizontalDoorA`/`B`, unused on this level) or in every other
   mechanism tested (`o_c_smalldoora`, `o_a_doorpairslide`, `o_d_door1`,
   `o_f_doorbig/doorsmall/gatebig`, `o_a_fan`). Extensive testing points to
   something specific to these two files' own source data, not a parsing
   or animation bug in the add-on — but this hasn't been root-caused. If
   you hit it: it's known, not a regression, and there's currently no
   workaround beyond leaving those two doors as they import.

2. **Five legacy-format `pa_area_*.bin` files still fail to parse.**
   Seven of twelve orphaned legacy-container files parse cleanly via the
   legacy path; five have a different internal object layout that hasn't
   been decoded. None of the five are listed in any level's `.lvt`
   manifest, so the game itself doesn't load them either — low priority.

3. **A third, unidentified `.bin` header** (`0xDEADDEAD` fill, seen on a
   few `visgeo*.bin` files) is detected and skipped cleanly, not decoded.
   Also absent from every `.lvt` manifest checked.

4. **`.lgc` is not decoded and not used.** It's a family of 44-byte
   records; the best-supported (not proven) hypothesis is an AI
   navigation/waypoint graph, based on records clustering spatially and
   one cluster climbing in height like a staircase. It doesn't correlate
   with region, light, or actor data from the corresponding `.lvt`. Left
   alone — geometry import doesn't need it.

5. **Chrome/reflective materials** (`Shader_ReflectionMap1/2` — Jango's
   armor, mirrors) are approximated with Metallic/Roughness rather than a
   true environment-projected reflection. The `EnvironmentMap` texture
   named in these `.mat` files isn't wired in; this engine's own UV
   projection convention for reflections was never checked against
   Blender's.

6. **`Shader_AdditiveUnlit` and `Shader_ModulateUnlit`** are approximated
   with alpha-blend, since Blender has no simple cross-version "Add" or
   "Multiply" `blend_method` to target generically. Visually close for
   sparse effects (neon signs, stars) but not a physically exact match.

7. **Skydomes are intentionally small** relative to level scale (a
   camera-relative rendering technique in the original engine) — this is
   not a scale bug and the add-on does not auto-scale them. Don't "fix" a
   skydome that looks tiny next to the level.

8. **Rigid-mechanism rest-pose/animation only resolves the first tracked
   ancestor** in a node's parent chain. Every mechanism seen has at most
   one meaningful tracked ancestor, so this hasn't been a practical
   limitation yet, but a mechanism with two levels of independently
   animated parents hasn't been tested.

9. **A few mechanisms have a matching `.abin` but no usable track**
   (seen: `O_B_Airspeeder01N`, `O_C_CylinderSign`) — their animation file's
   track name(s) don't match any node name, and there's more than one
   track so the single-track fallback (see next point) doesn't apply
   either. These import at their raw, static mesh position; a
   console message flags exactly this when it happens.

10. **Track-name mismatches are only resolved when unambiguous.** Some
    mechanisms (e.g. `o_a_fan`) name their only animation track something
    that matches no node at all (`"switch1"` vs. node names `"Spin"` /
    `"O_A_Fan"`). When there's *exactly one* track in the file, it's
    remapped onto the root node as a safe fallback. With two or more
    mismatched tracks, there's no reliable way to guess which track
    belongs to which node, so nothing is remapped.

---

## 6. Credits

- Every format in §3 was reverse-engineered directly from *Star Wars:
  Bounty Hunter*'s own shipped files — no format specification existed
  going in.
