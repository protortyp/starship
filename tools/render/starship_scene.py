"""Build and render a photoreal Starship landing in Blender from an exported trajectory.

    blender -b -P tools/render/starship_scene.py -- --track out/landing.json --time 41 \
        --camera ground --resolution 3840x2160 --samples 512 --out out/render/hero_ground.png

Everything is built procedurally at run time: the hull as a surface of revolution of the
simulator's profile with ring welds and panel seams in the shader, thousands of instanced
hexagonal heat tiles on the windward side (hull, nose and flap faces), flaps with hinge
fairings, the engine bay with three gimballing sea-level Raptors and three vacuum bells,
volumetric plumes with shock diamonds, a volumetric impingement splash and a procedural dust
ring that grows once the plume reaches the pad, a worn concrete pad, coastal terrain, ocean,
a Poly Haven dusk sky and compositor haze. Ship, flaps, gimbals, plume power and dust are
keyframed from the JSON written by `python -m starship.render.export`.

Modes: `still` renders one frame at `--time` seconds, `clip` renders `--start`..`--end` at
`--clip-fps`, `blend` only saves a .blend (with --save-blend) for inspection.
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector, noise

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contextlib
import itertools

from assets import fetch_assets

RING_HEIGHT = 1.83  # m, the steel rings the hull is welded from
PLATES_PER_RING = 4
TILE_ACROSS = 0.30  # m across flats
TILE_GAP = 0.016
TILE_THICK = 0.04
TILE_SECTOR = math.radians(100.0)  # half-angle of the tiled windward sector around +b3
PLUME_LENGTH = 48.0
DUST_START_ALTITUDE = 42.0  # plume reaches the pad below this engine-plane altitude


# ----------------------------------------------------------------------------------------------
# Arguments and scene setup
# ----------------------------------------------------------------------------------------------


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--track", required=True, help="JSON from starship.render.export")
    p.add_argument("--assets", default="out/render/assets", help="Poly Haven download dir")
    p.add_argument("--out", default="out/render/hero.png", help="still path or clip frame dir")
    p.add_argument("--mode", choices=["still", "clip", "blend"], default="still")
    p.add_argument("--time", type=float, default=41.0, help="sim seconds of the still")
    p.add_argument("--start", type=float, default=38.0, help="clip start, sim seconds")
    p.add_argument("--end", type=float, default=42.0, help="clip end, sim seconds")
    p.add_argument(
        "--clip-fps", type=int, default=30, help="output rate; a divisor of the track fps"
    )
    p.add_argument("--resolution", default="3840x2160")
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--camera", choices=["ground", "pad", "aerial", "wide"], default="ground")
    p.add_argument(
        "--camera-azimuth",
        default="auto",
        help="deg around the pad, or auto: belly azimuth at --time plus --camera-offset",
    )
    p.add_argument(
        "--camera-offset", type=float, default=100.0, help="deg from the belly direction for auto"
    )
    p.add_argument("--sun-offset", type=float, default=140.0, help="sun azimuth - camera azimuth")
    p.add_argument("--engines", choices=["sim", "all"], default="sim", help="all: light all three")
    p.add_argument("--device", choices=["OPTIX", "CUDA", "METAL", "CPU"], default="OPTIX")
    p.add_argument("--exposure", type=float, default=0.0)
    p.add_argument("--hdri", default="belfast_sunset_puresky", help="Poly Haven pure-sky HDRI id")
    p.add_argument("--haze", type=float, default=0.3, help="compositor mist strength")
    p.add_argument("--preview", action="store_true", help="coarser volumes, cheaper settings")
    p.add_argument("--max-tiles", type=int, default=0, help="cap the tile count (0 = all)")
    p.add_argument("--save-blend", default=None)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args(argv)


def log(msg):
    print(f"[starship_scene {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    return scene


# ----------------------------------------------------------------------------------------------
# Mesh helpers
# ----------------------------------------------------------------------------------------------


def make_object(
    name, verts, faces, materials=(), smooth=False, parent=None, location=None, recalc=False
):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    if recalc:
        # Closed shells assembled from several parts: make every normal point outwards.
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
    me.update()
    me.validate()
    if smooth:
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    for m in materials:
        me.materials.append(m)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    if parent is not None:
        ob.parent = parent
    if location is not None:
        ob.location = location
    return ob


def make_empty(name, parent=None, location=(0, 0, 0)):
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_size = 1.0
    bpy.context.scene.collection.objects.link(ob)
    if parent is not None:
        ob.parent = parent
    ob.location = location
    return ob


def revolve(profile, segments, axis="x", flip=False):
    """Surface of revolution of (x, r) points about local x, (x, r sin a, r cos a), or about
    local z with the same cyclic handedness. Rings with r = 0 collapse into a pole vertex.
    Normals face outwards when the profile runs towards +x; `flip` reverses the winding."""
    verts, rings = [], []
    for x, r in profile:
        if r < 1e-6:
            rings.append([len(verts)])
            verts.append((x, 0.0, 0.0) if axis == "x" else (0.0, 0.0, x))
        else:
            start = len(verts)
            for j in range(segments):
                a = 2 * math.pi * j / segments
                p = (x, r * math.sin(a), r * math.cos(a))
                verts.append(p if axis == "x" else (p[1], p[2], p[0]))
            rings.append(list(range(start, start + segments)))
    faces = []
    for ra, rb in itertools.pairwise(rings):
        if len(ra) == 1 and len(rb) == 1:
            continue
        if len(ra) == 1:
            faces += [(ra[0], rb[j], rb[(j + 1) % segments]) for j in range(segments)]
        elif len(rb) == 1:
            faces += [(ra[(j + 1) % segments], ra[j], rb[0]) for j in range(segments)]
        else:
            faces += [
                (ra[j], rb[j], rb[(j + 1) % segments], ra[(j + 1) % segments])
                for j in range(segments)
            ]
    if flip:
        faces = [tuple(reversed(f)) for f in faces]
    return verts, faces


def box(center, size):
    cx, cy, cz = center
    sx, sy, sz = (s / 2 for s in size)
    v = [
        (cx - sx, cy - sy, cz - sz), (cx + sx, cy - sy, cz - sz),
        (cx + sx, cy + sy, cz - sz), (cx - sx, cy + sy, cz - sz),
        (cx - sx, cy - sy, cz + sz), (cx + sx, cy - sy, cz + sz),
        (cx + sx, cy + sy, cz + sz), (cx - sx, cy + sy, cz + sz),
    ]  # fmt: skip
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return v, f


def cylinder(p0, p1, radius, segments=24, cap=True):
    """Closed cylinder between two points in the same local frame."""
    p0, p1 = Vector(p0), Vector(p1)
    axis = (p1 - p0).normalized()
    ref = Vector((0, 0, 1)) if abs(axis.z) < 0.9 else Vector((1, 0, 0))
    u = axis.cross(ref).normalized()
    w = axis.cross(u)
    verts, faces = [], []
    for p in (p0, p1):
        for j in range(segments):
            a = 2 * math.pi * j / segments
            verts.append(tuple(p + radius * (math.cos(a) * u + math.sin(a) * w)))
    for j in range(segments):
        k = (j + 1) % segments
        faces.append((j, k, segments + k, segments + j))
    if cap:
        faces.append(tuple(reversed(range(segments))))
        faces.append(tuple(range(segments, 2 * segments)))
    return verts, faces


def merge(parts):
    """Concatenate (verts, faces) lists into one."""
    verts, faces = [], []
    for v, f in parts:
        off = len(verts)
        verts += list(v)
        faces += [tuple(i + off for i in face) for face in f]
    return verts, faces


def add_bevel(ob, width=0.15, segments=3, angle=0.7):
    mod = ob.modifiers.new("Bevel", "BEVEL")
    mod.width = width
    mod.segments = segments
    mod.limit_method = "ANGLE"
    mod.angle_limit = angle
    mod.harden_normals = False
    return mod


# ----------------------------------------------------------------------------------------------
# Node helpers
# ----------------------------------------------------------------------------------------------


class NB:
    """Shader-tree builder with socket-or-constant inputs and column layout."""

    def __init__(self, tree):
        self.nodes, self.links = tree.nodes, tree.links
        self.count = 0

    def new(self, kind, **props):
        n = self.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        n.location = (240 * (self.count % 12), -200 * (self.count // 12))
        self.count += 1
        return n

    def put(self, node, key, value):
        sock = node.inputs[key]
        if isinstance(value, bpy.types.NodeSocket):
            self.links.new(value, sock)
        elif value is not None:
            try:
                sock.default_value = value
            except (TypeError, AttributeError) as e:
                raise TypeError(f"{node.bl_idname}.{sock.name} <- {value!r}: {e}") from e

    def math(self, op, a, b=None, c=None, clamp=False):
        n = self.new("ShaderNodeMath", operation=op, use_clamp=clamp)
        self.put(n, 0, a)
        if b is not None:
            self.put(n, 1, b)
        if c is not None:
            self.put(n, 2, c)
        return n.outputs[0]

    def vmath(self, op, a, b=None, scale=None):
        n = self.new("ShaderNodeVectorMath", operation=op)
        self.put(n, 0, a)
        if b is not None:
            self.put(n, 1, b)
        if scale is not None:
            self.put(n, "Scale", scale)
        return n.outputs["Value" if op in ("LENGTH", "DOT_PRODUCT", "DISTANCE") else "Vector"]

    def sep(self, v):
        n = self.new("ShaderNodeSeparateXYZ")
        self.put(n, 0, v)
        return n.outputs[0], n.outputs[1], n.outputs[2]

    def comb(self, x, y, z):
        n = self.new("ShaderNodeCombineXYZ")
        self.put(n, 0, x)
        self.put(n, 1, y)
        self.put(n, 2, z)
        return n.outputs[0]

    def noise(self, vec, scale, detail=2.0, rough=0.5, distortion=0.0, w=None):
        n = self.new("ShaderNodeTexNoise", noise_dimensions="4D" if w is not None else "3D")
        self.put(n, "Vector", vec)
        self.put(n, "Scale", scale)
        self.put(n, "Detail", detail)
        self.put(n, "Roughness", rough)
        self.put(n, "Distortion", distortion)
        if w is not None:
            self.put(n, "W", w)
        return n.outputs["Fac"]

    def ramp(self, fac, stops, interpolation="LINEAR"):
        n = self.new("ShaderNodeValToRGB")
        n.color_ramp.interpolation = interpolation
        el = n.color_ramp.elements
        el[0].position, el[0].color = stops[0][0], (*stops[0][1], 1.0)
        el[1].position, el[1].color = stops[-1][0], (*stops[-1][1], 1.0)
        for pos, col in stops[1:-1]:
            e = el.new(pos)
            e.color = (*col, 1.0)
        self.put(n, 0, fac)
        return n.outputs["Color"]

    def maprange(self, v, a, b, c, d, interp="LINEAR", clamp=True):
        n = self.new("ShaderNodeMapRange", interpolation_type=interp, clamp=clamp)
        self.put(n, 0, v)
        self.put(n, 1, a)
        self.put(n, 2, b)
        self.put(n, 3, c)
        self.put(n, 4, d)
        return n.outputs[0]

    def smooth(self, v, a, b):
        """0 below a, 1 above b (a > b inverts)."""
        return self.maprange(v, a, b, 0.0, 1.0, "SMOOTHSTEP")

    def mix(self, fac, a, b, kind="FLOAT"):
        n = self.new("ShaderNodeMix", data_type=kind)
        self.put(n, 0, fac)
        if kind == "FLOAT":
            self.put(n, 2, a)
            self.put(n, 3, b)
            return n.outputs[0]
        if kind == "VECTOR":
            self.put(n, 4, a)
            self.put(n, 5, b)
            return n.outputs[1]
        self.put(n, 6, (*a, 1.0) if isinstance(a, tuple) and len(a) == 3 else a)
        self.put(n, 7, (*b, 1.0) if isinstance(b, tuple) and len(b) == 3 else b)
        return n.outputs[2]

    def attr(self, name, kind="GEOMETRY"):
        return self.new("ShaderNodeAttribute", attribute_name=name, attribute_type=kind)

    def coords(self):
        return self.new("ShaderNodeTexCoord")

    def geometry(self):
        return self.new("ShaderNodeNewGeometry")

    def image(self, path, vec, non_color=False, scale=None):
        img = bpy.data.images.load(str(Path(path).resolve()), check_existing=True)
        if non_color:
            img.colorspace_settings.name = "Non-Color"
        n = self.new("ShaderNodeTexImage", image=img)
        n.extension = "REPEAT"
        if scale is not None:
            vec = self.vmath("SCALE", vec, scale=scale)
        self.put(n, "Vector", vec)
        return n

    def principled(self, **inputs):
        n = self.new("ShaderNodeBsdfPrincipled")
        for k, v in inputs.items():
            self.put(n, k, v)
        return n

    def output(self, surface=None, volume=None, displacement=None):
        n = self.new("ShaderNodeOutputMaterial")
        if surface is not None:
            self.put(n, "Surface", surface)
        if volume is not None:
            self.put(n, "Volume", volume)
        if displacement is not None:
            self.put(n, "Displacement", displacement)
        return n

    def bump(self, height, strength=0.3, distance=0.01, normal=None):
        n = self.new("ShaderNodeBump")
        self.put(n, "Strength", strength)
        self.put(n, "Distance", distance)
        self.put(n, "Height", height)
        if normal is not None:
            self.put(n, "Normal", normal)
        return n.outputs["Normal"]


def new_material(name, volume=False):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    if volume:
        mat.blend_method = "BLEND"
    return mat


def rgb(r, g, b):
    return (r, g, b, 1.0)


# ----------------------------------------------------------------------------------------------
# Materials
# ----------------------------------------------------------------------------------------------


def steel_material(name="Stainless", soot=True, tiled_sector=True):
    mat = new_material(name)
    nb = NB(mat.node_tree)
    obj = nb.coords().outputs["Object"]
    x, y, z = nb.sep(obj)
    phi = nb.math("ARCTAN2", y, z)
    # Ring welds every RING_HEIGHT metres, vertical plate seams staggered per ring.
    rx = nb.math("DIVIDE", x, RING_HEIGHT)
    ring_i = nb.math("FLOOR", rx)
    fr = nb.math("FRACT", rx)
    dist_ring = nb.math(
        "MULTIPLY",
        nb.math("SUBTRACT", 0.5, nb.math("ABSOLUTE", nb.math("SUBTRACT", fr, 0.5))),
        RING_HEIGHT,
    )
    weld = nb.smooth(dist_ring, 0.035, 0.008)
    heat_band = nb.smooth(dist_ring, 0.35, 0.04)
    # Plate seams drift a little from ring to ring rather than lining up like bricks.
    stagger = nb.math("MULTIPLY", nb.noise(nb.comb(ring_i, 0.0, 0.0), 0.9, 0.0), 0.35)
    u = nb.math("ADD", nb.math("MULTIPLY", phi, PLATES_PER_RING / (2 * math.pi)), stagger)
    fu = nb.math("FRACT", u)
    dist_v = nb.math(
        "MULTIPLY",
        nb.math("SUBTRACT", 0.5, nb.math("ABSOLUTE", nb.math("SUBTRACT", fu, 0.5))),
        2 * math.pi * 4.5 / PLATES_PER_RING,
    )
    vseam = nb.math("MULTIPLY", nb.smooth(dist_v, 0.025, 0.005), 0.5)
    seam = nb.math("MAXIMUM", weld, vseam)
    # Oxide patterns: each plate has a different sheen, with finer grain on top.
    n_big = nb.noise(obj, 0.12, 3.0)
    n_mid = nb.noise(obj, 1.6, 4.0, 0.6)
    n_fine = nb.noise(obj, 14.0, 2.0)
    plate_id = nb.noise(nb.comb(ring_i, nb.math("FLOOR", u), 0.0), 0.7, 0.0)
    rough = nb.math(
        "ADD", nb.math("MULTIPLY_ADD", n_big, 0.14, 0.17), nb.math("MULTIPLY", n_mid, 0.08)
    )
    rough = nb.math("ADD", rough, nb.math("MULTIPLY", plate_id, 0.08))
    rough = nb.math("ADD", rough, nb.math("MULTIPLY", seam, 0.18))
    base = nb.mix(
        nb.math("MULTIPLY", plate_id, 0.6), (0.76, 0.75, 0.73), (0.84, 0.83, 0.80), "RGBA"
    )
    tint = nb.ramp(
        nb.math("MULTIPLY", n_mid, heat_band),
        [
            (0.0, (0.72, 0.70, 0.68)),
            (0.45, (0.74, 0.60, 0.42)),
            (0.75, (0.50, 0.52, 0.66)),
            (1.0, (0.62, 0.60, 0.62)),
        ],
    )
    col = nb.mix(nb.math("MULTIPLY", heat_band, 0.45), base, tint, "RGBA")
    col = nb.mix(nb.math("MULTIPLY", seam, 0.3), col, (0.4, 0.39, 0.38), "RGBA")
    # Streaky grime running along the axis (rain and venting), stronger low on the hull.
    streak = nb.noise(
        nb.comb(
            nb.math("MULTIPLY", x, 0.12), nb.math("MULTIPLY", y, 1.8), nb.math("MULTIPLY", z, 1.8)
        ),
        1.0,
        5.0,
        0.7,
    )
    grime = nb.math("MULTIPLY", nb.smooth(streak, 0.45, 0.75), nb.smooth(x, 40.0, 4.0))
    col = nb.mix(nb.math("MULTIPLY", grime, 0.35), col, (0.35, 0.33, 0.30), "RGBA")
    rough = nb.math("ADD", rough, nb.math("MULTIPLY", grime, 0.25))
    # Welds sit in shallow grooves; the sheets between them oil-can, which is what bends the
    # sky reflections on the real vehicle.
    waves = nb.noise(nb.comb(nb.math("MULTIPLY", x, 0.9), y, z), 0.55, 2.0, 0.4)
    height = nb.math("ADD", nb.math("MULTIPLY", seam, -0.25), nb.math("MULTIPLY", n_fine, 0.04))
    height = nb.math("ADD", height, nb.math("MULTIPLY", waves, 1.0))
    normal = nb.bump(height, 0.3, 0.03)
    tangent = nb.new("ShaderNodeTangent", direction_type="RADIAL", axis="X")
    metal = nb.principled(
        **{
            "Base Color": col,
            "Metallic": 1.0,
            "Roughness": rough,
            "Normal": normal,
            "Anisotropic": 0.45,
            "Tangent": tangent.outputs[0],
        }
    )
    shader = metal.outputs[0]
    if soot:
        # Soot climbs from the engine bay with streaks; the bottom ring is nearly black.
        soot_f = nb.math(
            "MULTIPLY", nb.smooth(x, 12.0, 0.5), nb.math("MULTIPLY_ADD", streak, 0.6, 0.5)
        )
        soot_f = nb.math("MAXIMUM", soot_f, nb.math("MULTIPLY", nb.smooth(x, 2.5, 0.0), 0.9))
        dark = nb.principled(
            **{
                "Base Color": rgb(0.035, 0.03, 0.028),
                "Metallic": 0.2,
                "Roughness": 0.85,
                "Normal": normal,
            }
        )
        mixn = nb.new("ShaderNodeMixShader")
        nb.put(mixn, 0, nb.math("MULTIPLY", soot_f, 0.92, clamp=True))
        nb.links.new(shader, mixn.inputs[1])
        nb.links.new(dark.outputs[0], mixn.inputs[2])
        shader = mixn.outputs[0]
    if tiled_sector:
        # Under the tiles the hull wears a dark felt underlayment, visible in gaps and holes.
        sector = nb.smooth(nb.math("ABSOLUTE", phi), TILE_SECTOR + 0.04, TILE_SECTOR - 0.02)
        felt = nb.principled(
            **{"Base Color": rgb(0.05, 0.045, 0.04), "Metallic": 0.0, "Roughness": 0.95}
        )
        mixs = nb.new("ShaderNodeMixShader")
        nb.put(mixs, 0, sector)
        nb.links.new(shader, mixs.inputs[1])
        nb.links.new(felt.outputs[0], mixs.inputs[2])
        shader = mixs.outputs[0]
    nb.output(surface=shader)
    return mat


def tile_material():
    mat = new_material("HeatTiles")
    nb = NB(mat.node_tree)
    rnd = nb.attr("tile_random").outputs["Fac"]
    obj = nb.coords().outputs["Object"]
    patch = nb.noise(obj, 0.25, 3.0)
    # Multiplying colours keeps the per-tile value while adding broad patchiness; the top of
    # the ramp is the odd freshly replaced, lighter tile.
    mult = nb.new("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
    nb.put(mult, 0, 1.0)
    nb.put(
        mult,
        6,
        nb.ramp(
            rnd,
            [
                (0.0, (0.022, 0.022, 0.026)),
                (0.45, (0.04, 0.038, 0.037)),
                (0.85, (0.065, 0.06, 0.055)),
                (0.996, (0.075, 0.07, 0.064)),
                (0.997, (0.12, 0.115, 0.11)),
                (1.0, (0.14, 0.135, 0.13)),
            ],
        ),
    )
    nb.put(mult, 7, nb.mix(patch, (0.8, 0.78, 0.75), (1.25, 1.2, 1.12), "RGBA"))
    col = mult.outputs[2]
    grain = nb.noise(obj, 60.0, 3.0)
    rough = nb.math(
        "MULTIPLY_ADD", nb.noise(obj, 9.0, 2.0), 0.3, nb.math("MULTIPLY_ADD", rnd, 0.15, 0.42)
    )
    normal = nb.bump(grain, 0.12, 0.004)
    bsdf = nb.principled(
        **{
            "Base Color": col,
            "Metallic": 0.0,
            "Roughness": rough,
            "Normal": normal,
            "Specular IOR Level": 0.35,
        }
    )
    nb.output(surface=bsdf.outputs[0])
    return mat


def bell_material():
    mat = new_material("RaptorBell")
    nb = NB(mat.node_tree)
    obj = nb.coords().outputs["Object"]
    x, y, z = nb.sep(obj)
    phi = nb.math("ARCTAN2", y, z)
    ribs = nb.math("MULTIPLY_ADD", nb.math("SINE", nb.math("MULTIPLY", phi, 170.0)), 0.5, 0.5)
    n = nb.noise(obj, 3.0, 4.0)
    # Heat colouring along the bell: dark lip, straw and blue bands, dark chamber.
    col = nb.ramp(
        nb.math("ADD", x, nb.math("MULTIPLY", n, 0.5)),
        [
            (-3.2, (0.12, 0.10, 0.09)),
            (-2.4, (0.32, 0.26, 0.2)),
            (-1.6, (0.3, 0.3, 0.38)),
            (-0.9, (0.22, 0.2, 0.19)),
            (0.5, (0.18, 0.18, 0.18)),
        ],
    )
    rough = nb.math("MULTIPLY_ADD", n, 0.25, 0.3)
    normal = nb.bump(nb.math("ADD", ribs, nb.math("MULTIPLY", n, 0.3)), 0.3, 0.004)
    bsdf = nb.principled(
        **{"Base Color": col, "Metallic": 0.9, "Roughness": rough, "Normal": normal}
    )
    # Looking up the bell, the inside glows with the burn: inner faces have normals that point
    # towards the engine axis.
    thrust = nb.attr("thrust", "OBJECT").outputs["Fac"]
    radial = nb.vmath("NORMALIZE", nb.comb(0.0, y, z))
    facing = nb.vmath("DOT_PRODUCT", nb.coords().outputs["Normal"], radial)
    inside = nb.smooth(facing, -0.1, -0.6)
    glow = nb.math(
        "MULTIPLY",
        nb.math("MULTIPLY", inside, nb.smooth(x, -2.9, -0.8)),
        nb.math("MULTIPLY", thrust, 25.0),
    )
    emit = nb.new("ShaderNodeEmission")
    nb.put(emit, "Color", rgb(1.0, 0.62, 0.3))
    nb.put(emit, "Strength", glow)
    add = nb.new("ShaderNodeAddShader")
    nb.links.new(bsdf.outputs[0], add.inputs[0])
    nb.links.new(emit.outputs[0], add.inputs[1])
    nb.output(surface=add.outputs[0])
    return mat


def dark_metal_material(name="DarkMetal", color=(0.2, 0.2, 0.21), rough=0.5):
    mat = new_material(name)
    nb = NB(mat.node_tree)
    obj = nb.coords().outputs["Object"]
    n = nb.noise(obj, 2.0, 3.0)
    bsdf = nb.principled(
        **{
            "Base Color": rgb(*color),
            "Metallic": 0.8,
            "Roughness": nb.math("MULTIPLY_ADD", n, 0.25, rough - 0.1),
        }
    )
    nb.output(surface=bsdf.outputs[0])
    return mat


def plume_material(preview=False):
    """Volumetric methalox plume: blue-white core with shock diamonds, turbulent orange fringe."""
    mat = new_material("Plume", volume=True)
    nb = NB(mat.node_tree)
    obj = nb.coords().outputs["Object"]
    x, y, z = nb.sep(obj)
    d = nb.math("MULTIPLY", x, -1.0)  # distance downstream of the nozzle exit
    r = nb.math("SQRT", nb.math("ADD", nb.math("MULTIPLY", y, y), nb.math("MULTIPLY", z, z)))
    thrust = nb.attr("thrust", "OBJECT").outputs["Fac"]
    on = nb.smooth(thrust, 0.02, 0.25)
    tfrac = nb.math("MULTIPLY_ADD", thrust, 0.6, 0.4)
    core_r = nb.math("MULTIPLY", nb.math("MULTIPLY_ADD", d, 0.03, 0.62), nb.math("SQRT", tfrac))
    rr = nb.math("DIVIDE", r, core_r)
    rr2 = nb.math("MULTIPLY", rr, rr)
    core = nb.math("EXPONENT", nb.math("MULTIPLY", rr2, -3.0))
    axis_w = nb.math("EXPONENT", nb.math("MULTIPLY", rr2, -9.0))
    period = nb.math("MULTIPLY", nb.math("SQRT", tfrac), 1.7)
    phase = nb.math("COSINE", nb.math("MULTIPLY", nb.math("DIVIDE", d, period), 2 * math.pi))
    diamonds = nb.math(
        "MULTIPLY", nb.math("POWER", nb.math("MULTIPLY_ADD", phase, 0.5, 0.5), 3.0), axis_w
    )
    decay = nb.math("EXPONENT", nb.math("MULTIPLY", d, -1.0 / 16.0))
    world_z = nb.sep(nb.geometry().outputs["Position"])[2]
    ground = nb.smooth(world_z, 0.8, 3.5)
    inside = nb.smooth(d, -2.4, -1.6)
    frame_w = nb.attr("plume_time", "OBJECT").outputs["Fac"]
    turb = nb.math(
        "MULTIPLY_ADD",
        nb.noise(
            nb.comb(
                nb.math("MULTIPLY", x, 0.5),
                nb.math("MULTIPLY", y, 2.0),
                nb.math("MULTIPLY", z, 2.0),
            ),
            1.0,
            3.0,
            0.6,
            w=frame_w,
        ),
        0.7,
        0.65,
    )
    fringe = nb.math(
        "MULTIPLY",
        nb.math("EXPONENT", nb.math("MULTIPLY", rr2, -0.8)),
        nb.math("MULTIPLY", nb.smooth(d, 1.0, 8.0), 0.12),
    )
    strength = nb.math(
        "MULTIPLY",
        nb.math("ADD", nb.math("MULTIPLY", core, 0.9), nb.math("MULTIPLY", diamonds, 3.5)),
        decay,
    )
    strength = nb.math("ADD", strength, fringe)
    strength = nb.math(
        "MULTIPLY",
        strength,
        nb.math("MULTIPLY", turb, nb.math("MULTIPLY", ground, nb.math("MULTIPLY", on, inside))),
    )
    strength = nb.math("MULTIPLY", strength, 55.0)
    col = nb.ramp(
        nb.math("MULTIPLY", diamonds, 1.3),
        [(0.0, (0.42, 0.58, 1.0)), (0.5, (0.85, 0.9, 1.0)), (1.0, (1.0, 0.86, 0.66))],
    )
    col = nb.mix(nb.smooth(rr, 0.9, 1.6), col, (1.0, 0.5, 0.2), "RGBA")
    emit = nb.new("ShaderNodeEmission")
    nb.put(emit, "Color", col)
    nb.put(emit, "Strength", strength)
    absorb = nb.new("ShaderNodeVolumeAbsorption")
    nb.put(absorb, "Color", rgb(0.6, 0.7, 0.9))
    nb.put(absorb, "Density", nb.math("MULTIPLY", core, nb.math("MULTIPLY", on, 0.12)))
    add = nb.new("ShaderNodeAddShader")
    nb.links.new(emit.outputs[0], add.inputs[0])
    nb.links.new(absorb.outputs[0], add.inputs[1])
    nb.output(volume=add.outputs[0])
    # Cycles steps procedural volumes at 1/10 of their bounds times this rate; the plume's
    # bounds are ~20 m, so 0.1 gives ~0.2 m steps, fine enough for the shock diamonds.
    mat.cycles.volume_step_rate = 0.1 if not preview else 0.2
    return mat


def splash_material(preview=False):
    """Hot gas spreading over the pad where the plumes impinge."""
    mat = new_material("Splash", volume=True)
    nb = NB(mat.node_tree)
    obj = nb.coords().outputs["Object"]
    x, y, z = nb.sep(obj)
    d = nb.math("SQRT", nb.math("ADD", nb.math("MULTIPLY", x, x), nb.math("MULTIPLY", y, y)))
    power = nb.attr("splash", "OBJECT").outputs["Fac"]
    t = nb.attr("plume_time", "OBJECT").outputs["Fac"]
    radial = nb.math(
        "EXPONENT", nb.math("MULTIPLY", nb.math("POWER", nb.math("DIVIDE", d, 7.0), 2.0), -1.0)
    )
    low = nb.smooth(z, 3.5, 0.3)
    streaks = nb.noise(
        nb.comb(
            nb.math("MULTIPLY", x, 0.25), nb.math("MULTIPLY", y, 0.25), nb.math("MULTIPLY", z, 0.8)
        ),
        1.0,
        4.0,
        0.6,
        w=nb.math("MULTIPLY", t, 2.0),
    )
    streaks = nb.math("MULTIPLY_ADD", streaks, 1.2, 0.2)
    dens = nb.math(
        "MULTIPLY", nb.math("MULTIPLY", radial, low), nb.math("MULTIPLY", streaks, power)
    )
    emit = nb.new("ShaderNodeEmission")
    nb.put(
        emit,
        "Color",
        nb.ramp(
            nb.math("DIVIDE", d, 10.0),
            [(0.0, (1.0, 0.75, 0.45)), (0.4, (1.0, 0.45, 0.15)), (1.0, (0.7, 0.2, 0.05))],
        ),
    )
    nb.put(emit, "Strength", nb.math("MULTIPLY", dens, 2.5))
    vol = nb.new("ShaderNodeVolumePrincipled")
    nb.put(vol, "Color", rgb(0.8, 0.7, 0.6))
    nb.put(vol, "Density", nb.math("MULTIPLY", dens, 0.08))
    nb.put(vol, "Anisotropy", 0.4)
    add = nb.new("ShaderNodeAddShader")
    nb.links.new(emit.outputs[0], add.inputs[0])
    nb.links.new(vol.outputs[0], add.inputs[1])
    nb.output(volume=add.outputs[0])
    mat.cycles.volume_step_rate = 0.08 if not preview else 0.16  # ~0.35 m in a 60 m domain
    return mat


def dust_material(preview=False):
    """Expanding ground-hugging dust ring, shaped by object props dust_r (front radius), dust_h."""
    mat = new_material("Dust", volume=True)
    nb = NB(mat.node_tree)
    obj = nb.coords().outputs["Object"]
    x, y, z = nb.sep(obj)
    d = nb.math("SQRT", nb.math("ADD", nb.math("MULTIPLY", x, x), nb.math("MULTIPLY", y, y)))
    rf = nb.attr("dust_r", "OBJECT").outputs["Fac"]
    hscale = nb.attr("dust_h", "OBJECT").outputs["Fac"]
    on = nb.attr("dust_on", "OBJECT").outputs["Fac"]
    t = nb.attr("plume_time", "OBJECT").outputs["Fac"]
    # Ring front at ~0.8 rf with a thinner inner haze; the centre is scoured clear.
    ring_c = nb.math("MULTIPLY", rf, 0.78)
    ring_w = nb.math("MULTIPLY_ADD", rf, 0.22, 2.0)
    rel = nb.math("DIVIDE", nb.math("SUBTRACT", d, ring_c), ring_w)
    ring = nb.math("EXPONENT", nb.math("MULTIPLY", nb.math("MULTIPLY", rel, rel), -1.0))
    inner = nb.math(
        "MULTIPLY", nb.smooth(d, nb.math("MULTIPLY", rf, 0.15), nb.math("MULTIPLY", rf, 0.5)), 0.3
    )
    outside = nb.smooth(d, nb.math("MULTIPLY", rf, 1.25), nb.math("MULTIPLY", rf, 0.95))
    shape_r = nb.math("MULTIPLY", nb.math("MAXIMUM", ring, inner), outside)
    big = nb.noise(obj, 0.06, 4.0, 0.55, 0.8, w=nb.math("MULTIPLY", t, 0.35))
    # Billow columns of very different heights give the cauliflower silhouette.
    billow = nb.math("POWER", nb.smooth(big, 0.28, 0.72), 1.4)
    top = nb.math(
        "MULTIPLY",
        nb.math("MULTIPLY_ADD", d, 0.25, 1.0),
        nb.math("MULTIPLY", hscale, nb.math("MULTIPLY_ADD", billow, 2.6, 0.25)),
    )
    vert = nb.smooth(z, top, nb.math("MULTIPLY", top, 0.45))
    base = nb.smooth(z, -1.0, 0.2)
    lumps = nb.math("MULTIPLY_ADD", billow, 0.8, 0.2)
    fine = nb.math(
        "MULTIPLY_ADD", nb.noise(obj, 0.3, 6.0, 0.7, 0.5, w=nb.math("MULTIPLY", t, 0.8)), 1.1, 0.3
    )
    dens = nb.math(
        "MULTIPLY",
        nb.math("MULTIPLY", shape_r, vert),
        nb.math("MULTIPLY", base, nb.math("MULTIPLY", lumps, fine)),
    )
    dens = nb.math("MULTIPLY", dens, nb.math("MULTIPLY", on, 0.08))
    vol = nb.new("ShaderNodeVolumePrincipled")
    nb.put(vol, "Color", rgb(0.52, 0.46, 0.39))
    nb.put(vol, "Density", dens)
    nb.put(vol, "Anisotropy", 0.25)
    nb.put(vol, "Absorption Color", rgb(0.55, 0.5, 0.45))
    nb.output(volume=vol.outputs[0])
    mat.cycles.volume_step_rate = 0.04 if not preview else 0.08  # ~0.6 m in a 220 m domain
    return mat


def concrete_material(paths):
    mat = new_material("PadConcrete")
    nb = NB(mat.node_tree)
    obj = nb.coords().outputs["Object"]
    uv = nb.vmath("SCALE", obj, scale=1.0 / 7.0)
    diff = nb.image(paths["Diffuse"], uv)
    rough = nb.image(paths["Rough"], uv, non_color=True)
    nor = nb.image(paths["nor_gl"], uv, non_color=True)
    x, y, _z = nb.sep(obj)
    d = nb.math("SQRT", nb.math("ADD", nb.math("MULTIPLY", x, x), nb.math("MULTIPLY", y, y)))
    n_big = nb.noise(obj, 0.05, 3.0)
    n_mid = nb.noise(obj, 0.4, 4.0, 0.6)
    # Scorch: a dark disc with noisy edges and radial streaks from earlier burns.
    ang = nb.math("ARCTAN2", y, x)
    rays = nb.noise(
        nb.comb(nb.math("MULTIPLY", ang, 6.0), nb.math("MULTIPLY", d, 0.08), 0.0), 1.0, 3.0
    )
    scorch = nb.math(
        "MULTIPLY",
        nb.smooth(d, nb.math("MULTIPLY_ADD", n_mid, 9.0, 14.0), 3.0),
        nb.math("MULTIPLY_ADD", rays, 0.7, 0.5),
    )
    scorch = nb.math("MAXIMUM", scorch, nb.math("MULTIPLY", nb.smooth(d, 6.0, 1.0), 0.8))
    # Painted landing ring and cross, worn by the burns.
    ring = nb.smooth(nb.math("ABSOLUTE", nb.math("SUBTRACT", d, 14.5)), 0.35, 0.25)
    cross = nb.math(
        "MAXIMUM",
        nb.smooth(nb.math("ABSOLUTE", x), 0.3, 0.2),
        nb.smooth(nb.math("ABSOLUTE", y), 0.3, 0.2),
    )
    cross = nb.math("MULTIPLY", cross, nb.smooth(d, 7.0, 6.0))
    paint = nb.math("MULTIPLY", nb.math("MAXIMUM", ring, cross), nb.smooth(n_mid, 0.3, 0.5))
    paint = nb.math("MULTIPLY", paint, nb.math("SUBTRACT", 1.0, nb.math("MULTIPLY", scorch, 0.7)))
    tone = nb.mix(n_big, (0.8, 0.8, 0.8), (1.15, 1.12, 1.08), "RGBA")
    mult = nb.new("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
    nb.put(mult, 0, 1.0)
    nb.links.new(diff.outputs["Color"], mult.inputs[6])
    nb.put(mult, 7, tone)
    col = nb.mix(paint, mult.outputs[2], (0.85, 0.82, 0.7), "RGBA")
    col = nb.mix(nb.math("MULTIPLY", scorch, 0.85, clamp=True), col, (0.04, 0.035, 0.03), "RGBA")
    rough_v = nb.math("MULTIPLY_ADD", rough.outputs["Color"], 0.8, 0.15)
    rough_v = nb.mix(scorch, rough_v, 0.95)
    nmap = nb.new("ShaderNodeNormalMap")
    nb.put(nmap, "Strength", 0.8)
    nb.links.new(nor.outputs["Color"], nmap.inputs["Color"])
    bsdf = nb.principled(
        **{
            "Base Color": col,
            "Roughness": rough_v,
            "Normal": nmap.outputs[0],
            "Specular IOR Level": 0.4,
        }
    )
    nb.output(surface=bsdf.outputs[0])
    return mat


def terrain_material(paths):
    mat = new_material("Terrain")
    nb = NB(mat.node_tree)
    obj = nb.coords().outputs["Object"]
    uv = nb.vmath("SCALE", obj, scale=1.0 / 9.0)
    diff = nb.image(paths["Diffuse"], uv)
    rough = nb.image(paths["Rough"], uv, non_color=True)
    nor = nb.image(paths["nor_gl"], uv, non_color=True)
    # Second, rotated copy at another scale hides the tiling; broad noise varies the tone.
    rot = nb.new("ShaderNodeMapping")
    nb.put(rot, "Vector", obj)
    nb.put(rot, "Rotation", (0.0, 0.0, 0.7))
    nb.put(rot, "Scale", (1 / 23.0, 1 / 23.0, 1 / 23.0))
    diff2 = nb.image(paths["Diffuse"], rot.outputs[0])
    blend = nb.noise(obj, 0.02, 3.0)
    col = nb.mix(
        nb.smooth(blend, 0.35, 0.65), diff.outputs["Color"], diff2.outputs["Color"], "RGBA"
    )
    tone = nb.mix(nb.noise(obj, 0.004, 3.0), (0.75, 0.74, 0.72), (1.1, 1.05, 0.95), "RGBA")
    mult = nb.new("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
    nb.put(mult, 0, 1.0)
    nb.links.new(col, mult.inputs[6])
    nb.put(mult, 7, tone)
    # Vegetation patches: dry grass tufts as a colour mix beyond the pad apron.
    x, y, _z = nb.sep(obj)
    d = nb.math("SQRT", nb.math("ADD", nb.math("MULTIPLY", x, x), nb.math("MULTIPLY", y, y)))
    veg = nb.math(
        "MULTIPLY", nb.smooth(nb.noise(obj, 0.03, 5.0, 0.6), 0.5, 0.7), nb.smooth(d, 80.0, 160.0)
    )
    col = nb.mix(nb.math("MULTIPLY", veg, 0.7), mult.outputs[2], (0.22, 0.2, 0.09), "RGBA")
    nmap = nb.new("ShaderNodeNormalMap")
    nb.put(nmap, "Strength", 0.7)
    nb.links.new(nor.outputs["Color"], nmap.inputs["Color"])
    bsdf = nb.principled(
        **{
            "Base Color": col,
            "Roughness": nb.math("MULTIPLY_ADD", rough.outputs["Color"], 0.6, 0.35),
            "Normal": nmap.outputs[0],
            "Specular IOR Level": 0.3,
        }
    )
    nb.output(surface=bsdf.outputs[0])
    return mat


def water_material():
    mat = new_material("Ocean")
    nb = NB(mat.node_tree)
    obj = nb.coords().outputs["Object"]
    swell = nb.noise(nb.vmath("MULTIPLY", obj, (1.0, 0.35, 1.0)), 0.012, 6.0, 0.6)
    chop = nb.noise(obj, 0.25, 5.0, 0.65)
    h = nb.math("ADD", nb.math("MULTIPLY", swell, 1.0), nb.math("MULTIPLY", chop, 0.12))
    normal = nb.bump(h, 0.35, 0.5)
    bsdf = nb.principled(
        **{"Base Color": rgb(0.012, 0.045, 0.06), "Roughness": 0.08, "IOR": 1.333, "Normal": normal}
    )
    nb.output(surface=bsdf.outputs[0])
    return mat


def sun_direction_of_hdri(path):
    """Unit direction of the brightest pixel of an equirectangular HDRI, Cycles convention."""
    img = bpy.data.images.load(str(Path(path).resolve()), check_existing=True)
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    lum = buf.reshape(h, w, 4)[::4, ::4, :3].sum(axis=2)
    j, i = np.unravel_index(int(np.argmax(lum)), lum.shape)
    u, v = (i * 4 + 0.5) / w, (j * 4 + 0.5) / h
    phi, theta = math.pi * (1 - 2 * u), math.pi * (1 - v)
    return Vector(
        (math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta))
    )


def setup_world(hdri_path, sun_azimuth_deg, strength=1.0):
    world = bpy.data.worlds.new("Sky")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes.clear()
    nb = NB(world.node_tree)
    sun = sun_direction_of_hdri(hdri_path)
    current = math.degrees(math.atan2(sun.y, sun.x))
    mapping = nb.new("ShaderNodeMapping")
    nb.put(mapping, "Vector", nb.coords().outputs["Generated"])
    # Rotating the lookup vector by theta moves the sky features by -theta.
    nb.put(mapping, "Rotation", (0.0, 0.0, math.radians(current - sun_azimuth_deg)))
    env = nb.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(Path(hdri_path).resolve()), check_existing=True)
    nb.put(env, "Vector", mapping.outputs[0])
    bg = nb.new("ShaderNodeBackground")
    nb.links.new(env.outputs[0], bg.inputs[0])
    nb.put(bg, "Strength", strength)
    out = nb.new("ShaderNodeOutputWorld")
    nb.links.new(bg.outputs[0], out.inputs[0])
    world.mist_settings.start = 150.0
    world.mist_settings.depth = 6000.0
    world.mist_settings.falloff = "QUADRATIC"
    elevation = math.degrees(math.asin(sun.z))
    log(f"sun elevation {elevation:.1f} deg, rotated to azimuth {sun_azimuth_deg % 360:.0f}")
    return world, sun


# ----------------------------------------------------------------------------------------------
# Vehicle
# ----------------------------------------------------------------------------------------------


class Hull:
    def __init__(self, vehicle):
        self.length = vehicle["length"]
        self.radius = vehicle["diameter"] / 2
        self.nose = vehicle["nose_length"]
        self.base = self.length - self.nose
        self.ogive = (self.nose**2 + self.radius**2) / (2 * self.radius)

    def r(self, x):
        if x <= self.base:
            return self.radius
        if x >= self.length:
            return 0.0
        return math.sqrt(max(self.ogive**2 - (x - self.base) ** 2, 0.0)) + self.radius - self.ogive

    def slope(self, x):
        if x <= self.base or x >= self.length:
            return 0.0
        u = x - self.base
        return -u / math.sqrt(max(self.ogive**2 - u * u, 1e-9))


def hull_profile(hull: Hull):
    xs = [0.0]
    x = RING_HEIGHT
    while x < hull.base - 0.01:
        xs.append(x)
        x += RING_HEIGHT
    xs.append(hull.base)
    for t in np.linspace(0.0, 1.0, 48)[1:]:
        xs.append(hull.base + hull.nose * (1 - (1 - t) ** 1.5))
    return [(x, hull.r(x)) for x in xs]


def build_hull(ship, hull, mats):
    segments = 192
    outer = hull_profile(hull)
    # Open skirt: inner wall up to the aft dome, which bulges down into the engine bay.
    inner = [
        (0.0, hull.radius - 0.06), (6.6, hull.radius - 0.06), (6.3, 3.9), (5.9, 3.2),
        (5.5, 2.4), (5.2, 1.5), (5.05, 0.7), (5.0, 0.0),
    ]  # fmt: skip
    verts, faces = revolve(outer, segments)
    ob = make_object("Hull", verts, faces, [mats["steel"]], smooth=True, parent=ship)
    iv, if_ = revolve(inner, segments, flip=True)  # faces the engine bay
    make_object("EngineBay", iv, if_, [mats["steel_bay"]], smooth=True, parent=ship)
    # Skirt top ring stiffener and the leeward raceway (cable conduit) along the tanks.
    ring_v, ring_f = revolve(
        [
            (7.25, hull.radius),
            (7.25, hull.radius + 0.09),
            (7.45, hull.radius + 0.09),
            (7.45, hull.radius),
        ],
        segments,
    )
    make_object("SkirtRing", ring_v, ring_f, [mats["steel_plain"]], smooth=True, parent=ship)
    race = cylinder(
        (7.6, 0.0, -(hull.radius + 0.1)), (hull.base + 0.6, 0.0, -(hull.radius + 0.1)), 0.28, 20
    )
    r_ob = make_object(
        "Raceway", *race, [mats["steel_plain"]], smooth=True, parent=ship, recalc=True
    )
    add_bevel(r_ob, 0.03, 2)
    return ob


def hex_prism(center, n, t1, t2, rnd_tilt=0.0):
    """Vertices and faces of one tile: a chamfered hexagonal prism sitting on the surface."""
    rad = (TILE_ACROSS - TILE_GAP) / 2 / math.cos(math.pi / 6)
    if rnd_tilt:
        ax = random.uniform(-rnd_tilt, rnd_tilt)
        ay = random.uniform(-rnd_tilt, rnd_tilt)
        n = (n + ax * t1 + ay * t2).normalized()
        t1 = (t1 - t1.dot(n) * n).normalized()
        t2 = n.cross(t1)
    lift = random.uniform(-0.004, 0.004)
    base = center + (0.004 + lift) * n
    top = base + TILE_THICK * n
    verts = []
    for ring_c, scale in ((base, 1.0), (top, 0.9)):
        for k in range(6):
            a = math.radians(30 + 60 * k)
            verts.append(ring_c + scale * rad * (math.cos(a) * t2 + math.sin(a) * t1))
    faces = [(k, (k + 1) % 6, 6 + (k + 1) % 6, 6 + k) for k in range(6)]
    faces.append((6, 7, 8, 9, 10, 11))
    return verts, faces


def hull_tile_frames(hull, flaps_geom):
    """Tile centres with surface frames over the windward sector of the hull and nose."""
    pitch = TILE_ACROSS
    row_pitch = pitch * math.sqrt(3) / 2
    frames = []
    x = 0.35
    row = 0
    while x < hull.length - 0.25:
        r = hull.r(x)
        if r > 0.45:
            arc = 2 * TILE_SECTOR * r
            count = int(arc / pitch)
            offset = 0.5 if row % 2 else 0.0
            slope = hull.slope(x)
            for j in range(count):
                phi = -TILE_SECTOR + (j + 0.5 + offset) * pitch / r
                if phi > TILE_SECTOR - 0.3 * pitch / r:
                    continue
                rd = Vector((0.0, math.sin(phi), math.cos(phi)))
                center = Vector((x, 0.0, 0.0)) + r * rd
                n = Vector((-slope, rd.y, rd.z)).normalized()
                t1 = Vector((1.0, slope * rd.y, slope * rd.z)).normalized()
                t2 = n.cross(t1)
                frames.append((center, n, t1, t2))
        x += row_pitch
        row += 1
    return frames


def point_in_quad(px, py, quad, margin):
    """Inside test with an inward margin for a convex quad given counter-clockwise."""
    for (ax, ay), (bx, by) in zip(quad, quad[1:] + quad[:1], strict=True):
        ex, ey = bx - ax, by - ay
        length = math.hypot(ex, ey)
        cross = ex * (py - ay) - ey * (px - ax)
        if cross / length < margin:
            return False
    return True


def flap_planform(f, nose_sign):
    """Counter-clockwise quad in the flap frame: x along the hinge, y outboard."""
    length = f["hinge_top"] - f["hinge_bottom"]
    span = f["span"]
    tip = 2 * f["area"] / span - length  # tip chord from the trapezoid area
    sweep = length - tip
    if nose_sign > 0:
        return [
            (-length / 2, 0.0),
            (length / 2, 0.0),
            (length / 2 - sweep, span),
            (-length / 2, span),
        ]
    return [(-length / 2, 0.0), (length / 2, 0.0), (length / 2, span), (-length / 2 + sweep, span)]


def flap_frame(f):
    """Object matrix of a flap: x along the hinge (towards the nose for starboard flaps,
    away for port so that +z is the windward face for both), y outboard, origin at the hinge."""
    axis, op, folded = Vector(f["axis"]), Vector(f["open"]), Vector(f["folded"])
    ex = axis if axis.cross(op).z > 0 else -axis
    ey = (op - op.dot(ex) * ex).normalized()
    ez = ex.cross(ey)
    m = Matrix.Identity(4)
    for i in range(3):
        m[i][0], m[i][1], m[i][2] = ex[i], ey[i], ez[i]
        m[i][3] = f["hinge"][i]
    # Folding takes the open span towards `folded`, i.e. leeward: negative about local x.
    sign = -1.0 if folded.dot(ez) < 0 else 1.0
    return m, ex.x, sign


def build_flap(ship, f, k, mats, tiles_out):
    m, nose_sign, fold_sign = flap_frame(f)
    quad = flap_planform(f, nose_sign)
    span = f["span"]
    root_t, tip_t = 0.75, 0.22
    verts = [(x, y, 0.0) for x, y in quad]
    verts += [(x, y, -(root_t + (tip_t - root_t) * y / span)) for x, y in quad]
    faces = [(0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    ob = make_object(f"Flap{k}", verts, faces, [mats["steel_flap"]], parent=ship)
    ob.matrix_parent_inverse = Matrix.Identity(4)
    ob.matrix_basis = m
    ob.rotation_mode = "XYZ"
    add_bevel(ob, 0.18, 4)
    # Tiles on the windward (+z) face, following the planform with a margin.
    pitch = TILE_ACROSS
    row_pitch = pitch * math.sqrt(3) / 2
    y = 0.25
    row = 0
    n, t1, t2 = Vector((0, 0, 1)), Vector((1, 0, 0)), Vector((0, 1, 0))
    while y < span:
        offset = 0.5 * pitch if row % 2 else 0.0
        xs = np.arange(-8.0, 8.0, pitch) + offset
        for x in xs:
            if point_in_quad(x, y, quad, 0.22):
                tiles_out.append((Vector((x, y, 0.0)), n, t1, t2))
        y += row_pitch
        row += 1
    # Hinge fairing on the hull, leeward of the hinge line.
    length = f["hinge_top"] - f["hinge_bottom"]
    fair = cylinder((-length / 2 - 0.4, 0.15, -0.35), (length / 2 + 0.4, 0.15, -0.35), 0.5, 20)
    fob = make_object(
        f"FlapFairing{k}", *fair, [mats["steel_plain"]], smooth=True, parent=ship, recalc=True
    )
    fob.matrix_parent_inverse = Matrix.Identity(4)
    fob.matrix_basis = m
    add_bevel(fob, 0.1, 3)
    return ob, fold_sign


def build_tiles(parent, frames, mats, name, tilt=0.02, max_tiles=0, missing=0.004):
    if max_tiles and len(frames) > max_tiles:
        frames = frames[:: max(1, len(frames) // max_tiles)]
    verts, faces, rnd_values = [], [], []
    for center, n, t1, t2 in frames:
        if random.random() < missing:
            continue
        v, f = hex_prism(center, n, t1, t2, tilt)
        off = len(verts)
        verts += v
        faces += [tuple(i + off for i in face) for face in f]
        rnd_values += [random.random()] * len(v)
    ob = make_object(name, verts, faces, [mats["tiles"]], parent=parent)
    attr = ob.data.attributes.new("tile_random", "FLOAT", "POINT")
    attr.data.foreach_set("value", rnd_values)
    return ob, len(frames)


def bell_profile(exit_r, throat_r, length, chamber_r, chamber_len, segments_n=28):
    """(x, r) from the exit (x = -length) up to the chamber top, x along the engine axis."""
    prof = []
    for t in np.linspace(0.0, 1.0, segments_n):
        prof.append((-length + t * length, throat_r + (exit_r - throat_r) * (1 - t) ** 0.72))
    prof += [
        (0.15, chamber_r * 0.8),
        (0.4, chamber_r),
        (chamber_len, chamber_r),
        (chamber_len + 0.25, chamber_r * 0.6),
        (chamber_len + 0.3, 0.0),
    ]
    return prof


def build_engine(ship, engine, k, mats, preview):
    """Gimballing sea-level Raptor: empty at the pivot, bell and powerhead below/above it."""
    az, radius, pivot = engine["azimuth"], engine["radius"], engine["pivot_station"]
    emp = make_empty(f"Engine{k}", ship, (pivot, radius * math.sin(az), radius * math.cos(az)))
    emp.rotation_mode = "XYZ"
    exit_r = engine["exit_diameter"] / 2
    outer = bell_profile(exit_r, 0.27, pivot, 0.4, 1.0)
    inner = [(x + 0.02 if x < -0.3 else x, max(r - 0.035, 0.0)) for x, r in reversed(outer[:-5])]
    verts, faces = revolve([*outer, (-pivot + 0.02, exit_r - 0.035), *inner], 96)
    bell = make_object(
        f"Bell{k}", verts, faces, [mats["bell"]], smooth=True, parent=emp, recalc=True
    )
    bell["thrust"] = 0.0
    # Powerhead: two turbopumps, a manifold ring and feed lines.
    parts = [
        cylinder((0.5, 0.42, 0.1), (1.9, 0.62, 0.3), 0.26, 20),
        cylinder((0.5, -0.42, 0.1), (1.9, -0.62, 0.3), 0.26, 20),
        cylinder((0.6, 0.0, -0.45), (1.8, 0.0, -0.7), 0.2, 20),
        cylinder((-0.2, 0.0, 0.0), (1.3, 0.0, 0.0), 0.33, 24),
        cylinder((1.2, 0.5, 0.2), (1.2, -0.5, 0.2), 0.08, 12),
        cylinder((0.9, 0.3, -0.5), (1.9, 0.6, 0.3), 0.07, 12),
        cylinder((0.9, -0.3, -0.5), (1.9, -0.6, 0.3), 0.07, 12),
    ]
    ring_v, ring_f = revolve([(-0.35, 0.44), (-0.35, 0.56), (-0.2, 0.56), (-0.2, 0.44)], 48)
    parts.append((ring_v, ring_f))
    head = make_object(
        f"Powerhead{k}", *merge(parts), [mats["dark_metal"]], smooth=True, parent=emp, recalc=True
    )
    add_bevel(head, 0.02, 2)
    # Plume, from inside the bell down to the pad; origin at the nozzle exit.
    plen = PLUME_LENGTH
    prof = [(-d, exit_r * 1.05 + 0.08 * d) for d in np.linspace(plen, 2.0, 10)] + [
        (0.0, exit_r * 1.05),
        (2.3, 0.2),
    ]
    pv, pf = revolve(prof, 48)
    plume = make_object(
        f"Plume{k}", pv, pf, [mats["plume"]], smooth=True, parent=emp, location=(-pivot, 0.0, 0.0)
    )
    plume["thrust"] = 0.0
    plume["plume_time"] = 0.0
    plume.visible_shadow = False
    # Light cast by the exhaust (volume emission is not sampled as a light source).
    ldata = bpy.data.lights.new(f"PlumeLight{k}", "SPOT")
    ldata.energy = 0.0
    ldata.color = (0.78, 0.86, 1.0)
    ldata.spot_size = math.radians(70.0)
    ldata.spot_blend = 0.6
    ldata.shadow_soft_size = 1.0
    light = bpy.data.objects.new(f"PlumeLight{k}", ldata)
    bpy.context.scene.collection.objects.link(light)
    light.parent = emp
    light.location = (-pivot - 1.5, 0.0, 0.0)
    light.rotation_euler = (0.0, math.pi / 2, 0.0)
    return emp, bell, plume, light


def build_vacuum_engines(ship, hull, mats):
    for k, az in enumerate((math.pi / 3, math.pi, 5 * math.pi / 3)):
        r_pos = hull.radius - 1.3
        outer = bell_profile(1.15, 0.36, 3.8, 0.45, 1.0, 24)
        inner = [(x + 0.03, max(r - 0.04, 0.0)) for x, r in reversed(outer[:-5])]
        verts, faces = revolve([*outer, (-3.8 + 0.03, 1.15 - 0.04), *inner], 96)
        ob = make_object(
            f"RVac{k}",
            verts,
            faces,
            [mats["bell"]],
            smooth=True,
            parent=ship,
            location=(4.2, r_pos * math.sin(az), r_pos * math.cos(az)),
            recalc=True,
        )
        ob["thrust"] = 0.0


def build_ship(track, mats, preview, max_tiles):
    vehicle = track["vehicle"]
    hull = Hull(vehicle)
    ship = make_empty("Ship")
    ship.rotation_mode = "QUATERNION"
    build_hull(ship, hull, mats)
    flap_tiles = []
    flaps = []
    for k, f in enumerate(vehicle["flaps"]):
        tiles_local = []
        ob, sign = build_flap(ship, f, k, mats, tiles_local)
        tob, _ = build_tiles(
            ob, tiles_local, mats, f"FlapTiles{k}", max_tiles=max_tiles // 8 if max_tiles else 0
        )
        flaps.append((ob, sign))
        flap_tiles.append(tob)
    frames = hull_tile_frames(hull, vehicle["flaps"])
    _, n_tiles = build_tiles(ship, frames, mats, "HullTiles", max_tiles=max_tiles)
    log(f"hull tiles: {n_tiles}, flap tiles: {sum(len(t.data.polygons) // 7 for t in flap_tiles)}")
    engines = [build_engine(ship, e, k, mats, preview) for k, e in enumerate(vehicle["engines"])]
    build_vacuum_engines(ship, hull, mats)
    return ship, flaps, engines


# ----------------------------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------------------------


def build_pad(mats):
    size, thick = 58.0, 0.9
    v, f = box((0.0, 0.0, -thick / 2 + 0.001), (size, size, thick))
    pad = make_object("Pad", v, f, [mats["concrete"]])
    add_bevel(pad, 0.12, 3)
    # Apron: a wider, lower concrete ring, and a curb.
    v, f = box((0.0, 0.0, -0.3), (size + 40.0, size + 40.0, 0.5))
    apron = make_object("Apron", v, f, [mats["concrete"]])
    add_bevel(apron, 0.1, 2)
    for sx in (-1, 1):
        for sy in (-1, 1):
            pole = cylinder((sx * 72.0, sy * 72.0, -0.2), (sx * 72.0, sy * 72.0, 24.0), 0.3, 16)
            head = box((sx * 72.0, sy * 72.0, 24.3), (1.6, 1.6, 0.8))
            make_object(f"Mast{sx}{sy}", *merge([pole, head]), [mats["dark_metal"]], smooth=False)
    # Tank farm and a shelter for scale.
    for i in range(3):
        tank = cylinder((-170.0 + 14 * i, 150.0, 0.0), (-170.0 + 14 * i, 150.0, 12.0), 4.5, 32)
        make_object(f"Tank{i}", *tank, [mats["steel_plain"]], smooth=True)
    make_object("Shelter", *box((160.0, -120.0, 3.0), (18.0, 10.0, 6.0)), [mats["dark_metal"]])
    return pad


def build_terrain(mats, seed):
    half, n = 2600.0, 160
    step = 2 * half / n
    verts, faces = [], []
    rnd = Vector((seed * 3.1, seed * 1.7, 0.0))
    for j in range(n + 1):
        for i in range(n + 1):
            x, y = -half + i * step, -half + j * step
            d = math.hypot(x, y)
            flat = min(max((d - 90.0) / 300.0, 0.0), 1.0)
            p = Vector((x, y, 0.0)) * 0.0025 + rnd
            z = (noise.noise(p) * 2.2 + noise.noise(p * 4.0) * 0.5) * flat * flat
            z -= max(0.0, x - 700.0) * 0.012  # shelves down into the sea
            verts.append((x, y, z - 0.05))
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            faces.append((a, a + 1, a + n + 2, a + n + 1))
    ter = make_object("Terrain", verts, faces, [mats["terrain"]], smooth=True)
    sea = make_object("Ocean", *box((7000.0, 0.0, -0.6), (14000.0, 30000.0, 0.4)), [mats["water"]])
    return ter, sea


def build_ground_volumes(mats):
    sv, sf = revolve([(-0.2, 0.0), (-0.2, 30.0), (6.0, 30.0), (6.0, 0.0)], 24, axis="z")
    splash = make_object("Splash", sv, sf, [mats["splash"]])
    splash["splash"] = 0.0
    splash["plume_time"] = 0.0
    dv, df = revolve([(-0.5, 0.0), (-0.5, 110.0), (40.0, 110.0), (40.0, 0.0)], 24, axis="z")
    dust = make_object("Dust", dv, df, [mats["dust"]])
    for key in ("dust_r", "dust_h", "dust_on", "plume_time"):
        dust[key] = 0.0
    for ob in (splash, dust):
        ob.visible_shadow = False
    ldata = bpy.data.lights.new("SplashLight", "POINT")
    ldata.energy = 0.0
    ldata.color = (1.0, 0.55, 0.25)
    ldata.shadow_soft_size = 4.0
    light = bpy.data.objects.new("SplashLight", ldata)
    bpy.context.scene.collection.objects.link(light)
    light.location = (0.0, 0.0, 2.5)
    return splash, dust, light


def belly_azimuth(track, t):
    """World azimuth (deg) the belly (+b3) faces at sim time t."""
    fr = track["frames"]
    i = min(round(t * track["meta"]["fps"]), len(fr["time"]) - 1)
    x, y, z, w = fr["quaternion"][i]
    bx = 2 * (x * z + w * y)
    by = 2 * (y * z - w * x)
    return math.degrees(math.atan2(by, bx))


def camera_azimuth(args, track):
    if args.camera_azimuth != "auto":
        return float(args.camera_azimuth)
    t = args.time if args.mode != "clip" else 0.5 * (args.start + args.end)
    return belly_azimuth(track, t) + args.camera_offset


def build_camera(args, aim, azimuth_deg):
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    az = math.radians(azimuth_deg)
    presets = {
        # distance, height, focal length, f-stop, aim height above the engine plane
        "ground": (280.0, 2.5, 75.0, 4.0, 18.0),
        "pad": (85.0, 1.7, 24.0, 5.6, 14.0),
        "aerial": (260.0, 120.0, 60.0, 5.6, 22.0),
        "wide": (520.0, 1.8, 40.0, 8.0, 18.0),
    }
    dist, height, focal, fstop, aim_h = presets[args.camera]
    cam.location = (dist * math.cos(az), dist * math.sin(az), height)
    cam_data.lens = focal
    cam_data.sensor_width = 36.0
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.clip_end = 50000.0
    cam_data.dof.use_dof = not args.preview
    cam_data.dof.focus_object = aim
    cam_data.dof.aperture_fstop = fstop
    cam_data.dof.aperture_blades = 9
    track = cam.constraints.new("TRACK_TO")
    track.target = aim
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    return cam, aim_h


# ----------------------------------------------------------------------------------------------
# Animation
# ----------------------------------------------------------------------------------------------


def dust_front(age):
    """Front radius (m) of the dust ring `age` seconds after the plume first hit the pad."""
    if age <= 0:
        return 0.0
    return 18.0 * (1 - math.exp(-age / 2.2)) + 5.0 * age


def animate(track, args, ship, flaps, engines, splash, dust, splash_light, aim, aim_h, frames):
    fr = track["frames"]
    fps = track["meta"]["fps"]
    pos = np.array(fr["position"])
    quat = np.array(fr["quaternion"])
    flap_ang = np.array(fr["flaps"])
    thrust = np.array(fr["thrust"])
    g2, g3 = np.array(fr["gimbal2"]), np.array(fr["gimbal3"])
    alt = np.array(fr["altitude"])
    if args.engines == "all":
        thrust = np.maximum(thrust, thrust.max(axis=1, keepdims=True))
    # Dust starts once a lit plume reaches the pad.
    lit = thrust.max(axis=1) > 0.2
    below = lit & (alt < DUST_START_ALTITUDE)
    first = int(np.argmax(below)) if below.any() else len(alt)
    for i in frames:
        i = min(max(i, 0), len(alt) - 1)
        t = i / fps
        ship.location = tuple(pos[i])
        x, y, z, w = quat[i]
        ship.rotation_quaternion = (w, x, y, z)
        ship.keyframe_insert("location", frame=i)
        ship.keyframe_insert("rotation_quaternion", frame=i)
        for k, (ob, sign) in enumerate(flaps):
            ob.rotation_euler = (sign * flap_ang[i, k], 0.0, 0.0)
            ob.keyframe_insert("rotation_euler", frame=i)
        for k, (emp, bell, plume, light) in enumerate(engines):
            emp.rotation_euler = (0.0, -g3[i, k], g2[i, k])
            emp.keyframe_insert("rotation_euler", frame=i)
            for ob in (bell, plume):
                ob["thrust"] = float(thrust[i, k])
                ob.keyframe_insert('["thrust"]', frame=i)
            plume["plume_time"] = t
            plume.keyframe_insert('["plume_time"]', frame=i)
            light.data.energy = float(thrust[i, k]) * 2.2e6
            light.data.keyframe_insert("energy", frame=i)
        total = float(thrust.sum(axis=1)[i])
        reach = min(max((DUST_START_ALTITUDE - alt[i]) / 18.0, 0.0), 1.0)
        power = total * reach
        for ob in (splash, dust, splash_light):
            ob.location = (pos[i, 0], pos[i, 1], ob.location.z)
            ob.keyframe_insert("location", frame=i)
        splash["splash"] = power
        splash["plume_time"] = t
        splash.keyframe_insert('["splash"]', frame=i)
        splash.keyframe_insert('["plume_time"]', frame=i)
        splash_light.data.energy = power * 6e5
        splash_light.data.keyframe_insert("energy", frame=i)
        age = max(0.0, (i - first) / fps)
        dust["dust_r"] = dust_front(age)
        dust["dust_h"] = 0.7 + 0.25 * age
        dust["dust_on"] = min(age / 1.5, 1.0) * (0.6 + 0.4 * min(total / 0.8, 1.0))
        dust["plume_time"] = t
        for key in ("dust_r", "dust_h", "dust_on", "plume_time"):
            dust.keyframe_insert(f'["{key}"]', frame=i)
        aim.location = (pos[i, 0], pos[i, 1], pos[i, 2] + aim_h)
        aim.keyframe_insert("location", frame=i)
    # Linear interpolation between the dense keys avoids overshoot in the motion blur.
    for ob in [ship, aim, splash, dust, *[f for f, _ in flaps], *[e for e, *_ in engines]]:
        if ob.animation_data and ob.animation_data.action:
            for fc in ob.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"


# ----------------------------------------------------------------------------------------------
# Render settings
# ----------------------------------------------------------------------------------------------


def configure_render(scene, args, fps):
    scene.render.engine = "CYCLES"
    scene.render.fps = round(fps)
    w, h = (int(v) for v in args.resolution.lower().split("x"))
    scene.render.resolution_x, scene.render.resolution_y = w, h
    scene.render.resolution_percentage = 100
    cy = scene.cycles
    cy.samples = args.samples
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.03 if args.preview else 0.008
    cy.max_bounces = 8
    cy.diffuse_bounces = 3
    cy.glossy_bounces = 4
    cy.transmission_bounces = 6
    cy.volume_bounces = 1
    cy.volume_step_rate = 2.0 if args.preview else 1.0
    cy.volume_max_steps = 256 if args.preview else 1024
    cy.volume_preview_step_rate = 4.0
    cy.caustics_reflective = False
    cy.caustics_refractive = False
    cy.blur_glossy = 1.0
    cy.sample_clamp_indirect = 20.0
    cy.use_light_tree = True
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.5
    scene.render.use_persistent_data = True
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        scene.view_settings.look = "None"
    scene.view_settings.exposure = args.exposure
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "16"
    scene.render.image_settings.compression = 20
    # Device: OptiX (denoiser included), CUDA/Metal with OpenImageDenoise, or CPU.
    device = args.device
    cy.device = "CPU"
    if device != "CPU":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        try:
            prefs.compute_device_type = device
        except TypeError:
            log(f"{device} unavailable, trying CUDA")
            device = "CUDA"
            try:
                prefs.compute_device_type = device
            except TypeError:
                device = "CPU"
        if device != "CPU":
            prefs.refresh_devices()
            used = 0
            for d in prefs.devices:
                d.use = d.type == device
                used += d.use
            if used:
                cy.device = "GPU"
                log(f"rendering on {used} {device} device(s)")
            else:
                log(f"no {device} devices found, using CPU")
                device = "CPU"
    cy.use_denoising = True
    cy.denoiser = "OPTIX" if device == "OPTIX" else "OPENIMAGEDENOISE"
    cy.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    cy.denoising_prefilter = "ACCURATE"
    if cy.denoiser == "OPENIMAGEDENOISE":
        cy.denoising_use_gpu = device != "CPU"
    bpy.context.view_layer.use_pass_mist = True
    return device


def setup_compositor(scene, haze=0.45):
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    rl = tree.nodes.new("CompositorNodeRLayers")
    glare = tree.nodes.new("CompositorNodeGlare")
    glare.glare_type = "FOG_GLOW"
    for name, val in (("quality", "HIGH"), ("size", 8), ("mix", -0.3), ("threshold", 2.0)):
        with contextlib.suppress(AttributeError, TypeError):
            setattr(glare, name, val)
    for name, val in (("Threshold", 2.0), ("Strength", 0.25), ("Size", 0.9)):
        sock = glare.inputs.get(name)
        if sock is not None and sock.type == "VALUE" and not sock.is_linked:
            sock.default_value = val
    tree.links.new(rl.outputs["Image"], glare.inputs["Image"])
    # Atmospheric haze: lift distant pixels towards the sky colour by the mist pass.
    scale = tree.nodes.new("CompositorNodeMath")
    scale.operation = "MULTIPLY"
    scale.inputs[1].default_value = haze
    tree.links.new(rl.outputs["Mist"], scale.inputs[0])
    mix = tree.nodes.new("CompositorNodeMixRGB")
    mix.blend_type = "MIX"
    mix.inputs[2].default_value = (0.66, 0.6, 0.58, 1.0)
    tree.links.new(scale.outputs[0], mix.inputs[0])
    tree.links.new(glare.outputs["Image"], mix.inputs[1])
    comp = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(mix.outputs[0], comp.inputs["Image"])


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------


def main():
    args = parse_args()
    random.seed(args.seed)
    t0 = time.time()
    track = json.loads(Path(args.track).read_text())
    fps = track["meta"]["fps"]
    n_frames = track["meta"]["frames"]
    log(f"track: {n_frames} frames at {fps} fps, status {track['meta']['status']}")
    scene = reset_scene()
    assets = fetch_assets(args.assets, hdri=args.hdri)
    hdri = assets[args.hdri]["hdr"]

    mats = {
        "steel": steel_material("Stainless"),
        "steel_flap": steel_material("StainlessFlap", soot=False, tiled_sector=False),
        "steel_plain": steel_material("StainlessPlain", soot=False, tiled_sector=False),
        "steel_bay": steel_material("StainlessBay", soot=True, tiled_sector=False),
        "tiles": tile_material(),
        "bell": bell_material(),
        "dark_metal": dark_metal_material(),
        "plume": plume_material(args.preview),
        "splash": splash_material(args.preview),
        "dust": dust_material(args.preview),
        "concrete": concrete_material(assets["concrete_floor_worn_001"]),
        "terrain": terrain_material(assets["coast_sand_rocks_02"]),
        "water": water_material(),
    }
    cam_az = camera_azimuth(args, track)
    setup_world(hdri, cam_az + args.sun_offset, strength=1.0)
    ship, flaps, engines = build_ship(track, mats, args.preview, args.max_tiles)
    build_pad(mats)
    build_terrain(mats, args.seed)
    splash, dust, splash_light = build_ground_volumes(mats)
    aim = make_empty("CameraAim")
    _cam, aim_h = build_camera(args, aim, cam_az)
    log(f"camera azimuth {cam_az:.0f} deg ({args.camera})")
    log(f"scene built in {time.time() - t0:.1f} s")

    if args.mode == "clip":
        start, end = round(args.start * fps), round(args.end * fps)
        # Frames are picked from the 60 fps track, so the output rate is fps / step.
        step = max(1, round(fps / args.clip_fps))
        if abs(fps / step - args.clip_fps) > 1e-6:
            log(f"clip fps {args.clip_fps} does not divide {fps:g}; rendering {fps / step:g} fps")
    else:
        start = end = round(args.time * fps)
        step = 1
    start, end = max(0, min(start, n_frames - 1)), max(0, min(end, n_frames - 1))
    animate(
        track,
        args,
        ship,
        flaps,
        engines,
        splash,
        dust,
        splash_light,
        aim,
        aim_h,
        range(start - 2, end + 3),
    )
    scene.frame_start, scene.frame_end, scene.frame_step = start, end, step
    scene.frame_set(start)
    device = configure_render(scene, args, fps)
    # A 180-degree shutter at the clip's output rate, not at the track's 60 fps.
    scene.render.motion_blur_shutter = 0.5 * step
    setup_compositor(scene, args.haze)
    log(f"frames {start}..{end} step {step}; {args.resolution} @ {args.samples} spp on {device}")

    out = Path(args.out)
    if args.save_blend:
        Path(args.save_blend).parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(
            filepath=str(Path(args.save_blend).resolve()), relative_remap=False
        )
        log(f"saved {args.save_blend}")
    if args.mode == "blend":
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    if args.mode == "still":
        scene.render.filepath = str(out.resolve())
        bpy.ops.render.render(write_still=True)
        log(f"rendered {out} in {time.time() - t1:.0f} s")
    else:
        out.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(out.resolve() / "frame_")
        bpy.ops.render.render(animation=True)
        n = (end - start) // step + 1
        dt = time.time() - t1
        log(f"rendered {n} frames to {out} in {dt:.0f} s ({dt / n:.0f} s/frame)")


if __name__ == "__main__":
    main()
