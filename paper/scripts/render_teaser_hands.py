"""Render the fist-cycle keyframes as a skinned hand posed by the measurements.

The mesh is Joerg Sla's limbKit (a paid Blender asset, kept outside this
repository). Its GN-limbKit modifier exposes three flexion angles per finger,
so each keyframe is posed by the interior angles of the measured landmark
chains: the joint angles are data, the hand itself is the kit's geometry and
proportions. One fixed orthographic camera and one scale for every pose.

Geometry comes from output/figures/teaser/hands.json, written by
scripts/make_teaser.py, so the frame selection lives in one place.

    .venv/bin/python scripts/make_teaser.py            # writes hands.json
    /Applications/Blender.app/Contents/MacOS/Blender -b -P scripts/render_teaser_hands.py
    .venv/bin/python scripts/make_teaser.py            # embeds the renders
"""
import hashlib
import json
from pathlib import Path
import sys

import bpy
import mathutils
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/figures/teaser"   # = make_teaser.py's paths.OUTPUT / "figures/teaser"; Blender's Python does not see scripts/
HANDS = OUT / "hands.json"
KIT = Path.home() / "Downloads/limbKit_02.blend"
SUBJECT, MODIFIER = "handStylized", "GN-limbKit"
# The palm side with the fingers up the frame: a closed fist shows its curled
# fingers and thumb wrap here, where the back of the hand hides both.
VIEW_EYE, VIEW_UP = (0.0, 0.0, -1.0), (0.0, -1.0, 0.0)
NAIL_EDGE, NAIL_STRENGTH = 0.14, 0.18   # Socket_52/68; the nail is painted, and a
# high edge position wraps it round onto the palm, so keep it small and faint.
WRIST_FADE = (0.05, 0.26)   # the arm is cut open at the wrist; dissolve the stump
PIXELS, SAMPLES = 760, 256

# GN-limbKit flexion sockets, in the order hands.json lists its chains. The
# thumb exposes no root bend, so its first measured angle has nowhere to go.
BENDS = [("thumb", None, "Socket_40", "Socket_39"),
         ("index", "Socket_43", "Socket_14", "Socket_13"),
         ("middle", "Socket_44", "Socket_20", "Socket_19"),
         ("ring", "Socket_46", "Socket_26", "Socket_25"),
         ("little", "Socket_45", "Socket_32", "Socket_31")]


def flexion(pose, chain):
    """Interior angles at the three finger joints, in radians; 0 is straight."""
    points = [mathutils.Vector(pose[i]) for i in chain]
    limbs = [b - a for a, b in zip(points[:-1], points[1:])]
    return [a.angle(b) for a, b in zip(limbs[:-1], limbs[1:])]


def main():
    hands = json.loads(HANDS.read_text())
    assert KIT.exists(), f"limbKit not found at {KIT}"
    bpy.ops.wm.open_mainfile(filepath=str(KIT))
    hand = bpy.data.objects[SUBJECT]
    kit = hand.modifiers[MODIFIER]
    for obj in bpy.data.objects:
        obj.hide_render = obj is not hand

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    scene.render.resolution_x = scene.render.resolution_y = PIXELS
    scene.render.resolution_percentage = 100   # the kit ships a scaled preset
    # The kit ships a video output preset, so the media type has to change first.
    scene.render.image_settings.media_type = "IMAGE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.5
    scene.world = world

    # Lights are placed relative to the view, so changing VIEW_EYE keeps a
    # conventional key/fill/rim instead of scattering them around the hand.
    eye = mathutils.Vector(VIEW_EYE).normalized()
    up = mathutils.Vector(VIEW_UP).normalized()
    across = up.cross(eye)

    def lamp(name, energy, direction, angle):
        light = bpy.data.lights.new(name, "SUN")
        light.energy, light.angle = energy, angle
        obj = bpy.data.objects.new(name, light)
        obj.rotation_euler = direction.normalized().to_track_quat("Z", "Y").to_euler()
        scene.collection.objects.link(obj)

    lamp("key", 4.0, eye * 0.75 + up * 0.80 - across * 0.70, 0.12)
    lamp("fill", 1.1, eye * 0.90 - up * 0.30 + across * 0.85, 0.60)
    lamp("rim", 1.6, -eye * 0.80 + up * 0.50 + across * 0.30, 0.30)
    kit["Socket_52"] = NAIL_EDGE
    kit["Socket_68"] = NAIL_STRENGTH

    poses = hands["poses_mm"]
    angles = [[flexion(pose, chain) for chain in hands["chains"]] for pose in poses]

    def apply(pose_angles):
        for measured, (_, root, centre, tip) in zip(pose_angles, BENDS):
            values = measured if root else measured[1:]
            for socket, value in zip([s for s in (root, centre, tip) if s], values):
                kit[socket] = float(value)
        hand.update_tag()
        bpy.context.view_layer.update()

    # One framing for every pose: the union of their evaluated bounding boxes.
    low = mathutils.Vector((1e9,) * 3)
    high = mathutils.Vector((-1e9,) * 3)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for pose_angles in angles:
        apply(pose_angles)
        depsgraph.update()
        evaluated = hand.evaluated_get(depsgraph)
        for corner in evaluated.bound_box:
            world_point = evaluated.matrix_world @ mathutils.Vector(corner)
            low = mathutils.Vector(map(min, low, world_point))
            high = mathutils.Vector(map(max, high, world_point))
    centre = (low + high) / 2
    extent = high - low
    tall = abs(extent.dot(up))
    wide = abs(extent.dot(across))
    reach = max(tall, wide)

    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 1.12 * reach
    basis = mathutils.Matrix((across, up, eye)).transposed().to_4x4()
    camera.matrix_world = mathutils.Matrix.Translation(centre + eye * 4 * reach) @ basis
    scene.collection.objects.link(camera)
    scene.camera = camera

    written = []
    for pose_angles, frame, posture in zip(angles, hands["frames_50hz"], hands["posture"]):
        apply(pose_angles)
        scene.render.filepath = str(OUT / "hands" / f"hand-{frame:03d}-{posture}")
        bpy.ops.render.render(write_still=True)
        # The mesh is an arm cut off at the wrist, which renders as a dark open
        # disc. Fade the bottom of the frame so it dissolves instead.
        saved = Path(scene.render.filepath).with_suffix(".png")
        image = bpy.data.images.load(str(saved))
        wide, tall_px = image.size
        pixels = np.array(image.pixels[:], dtype=np.float32).reshape(tall_px, wide, 4)
        rows = np.arange(tall_px) / tall_px    # Blender stores bottom row first.
        low, high = WRIST_FADE
        pixels[:, :, 3] *= np.clip((rows - low) / (high - low), 0, 1)[:, None]
        image.pixels.foreach_set(pixels.ravel())
        image.filepath_raw, image.file_format = str(saved), "PNG"
        image.save()
        bpy.data.images.remove(image)
        written.append(saved.name)

    # The panel embeds these only when this digest still matches its hands.json,
    # so an edited selection cannot silently keep the old renders.
    manifest = {"hands_json_sha256": hashlib.sha256(HANDS.read_bytes()).hexdigest(),
                "rendered": written, "pixels": PIXELS, "samples": SAMPLES,
                "engine": "Cycles", "blender": bpy.app.version_string,
                # The panel sizes each square render from these, so the hand
                # lands at the height the layout asks for.
                "frame_units": camera.data.ortho_scale,
                "hand_units": float(tall),
                "mesh": "limbKit_02 handStylized (Joerg Sla, limbKit); paid asset, "
                        "referenced outside the repository and not redistributed here",
                "mesh_sha256": hashlib.sha256(KIT.read_bytes()).hexdigest(),
                "posed_by": "interior angles of the measured landmark chains, written to "
                            "GN-limbKit's per-finger Bend Root/Center/Tip inputs",
                "not_posed": "the thumb's root flexion (the kit exposes no socket for it), "
                             "finger abduction, and wrist orientation",
                "view": {"eye": list(VIEW_EYE), "up": list(VIEW_UP),
                         "shows": "the palm side"},
                "nail_edge_position": NAIL_EDGE, "nail_edge_intensity": NAIL_STRENGTH,
                "wrist_fade_fraction": list(WRIST_FADE),
                "degrees": [[[round(a * 57.29577951308232, 1) for a in finger]
                             for finger in pose] for pose in angles]}
    (OUT / "hands" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "degrees"}, indent=2), file=sys.stderr)


main()
