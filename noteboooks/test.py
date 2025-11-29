#!/usr/bin/env python3
"""
generate_model_sdf.py

Auto-generate model.sdf listing each mesh in meshes/ as a separate link
with visual + collision geometry. Staggers each link along X by `spacing`.
"""

import os
from xml.etree import ElementTree as ET

MODEL_DIR = os.path.expanduser("~/.gazebo/models/gazebo_model")
MESH_DIR = os.path.join(MODEL_DIR, "meshes")
OUT_SDF = os.path.join(MODEL_DIR, "model.sdf")
SDF_VERSION = "1.6"   # change to 1.7 if you prefer
spacing = 1.0         # separation between links (meters)

# gather visual files
files = os.listdir(MESH_DIR)
visuals = sorted([f for f in files if f.endswith("_visual.obj")])

if len(visuals) == 0:
    print("No *_visual.obj files found in", MESH_DIR)
    raise SystemExit(1)

# XML root
sdf = ET.Element("sdf", version=SDF_VERSION)
model = ET.SubElement(sdf, "model", name="gazebo_model")
static = ET.SubElement(model, "static")
static.text = "true"

# iterate and make a link per visual
for i, vis in enumerate(visuals):
    # derive collision filename
    base = vis.replace("_visual.obj", "")
    coll = f"{base}_collision.obj"
    if coll not in files:
        print(f"WARNING: collision mesh not found for {vis} (expected {coll}). Skipping collision.")
        coll = None

    link = ET.SubElement(model, "link", name=f"{base}")
    pose = ET.SubElement(link, "pose")
    # place them offset in X so they don't overlap
    x = i * spacing
    pose.text = f"{x} 0 0 0 0 0"

    # visual
    visual = ET.SubElement(link, "visual", name=f"{base}_visual")
    geom_v = ET.SubElement(visual, "geometry")
    mesh_v = ET.SubElement(geom_v, "mesh")
    uri_v = ET.SubElement(mesh_v, "uri")
    uri_v.text = f"model://gazebo_model/meshes/{vis}"

    # collision (if exists)
    if coll:
        collision = ET.SubElement(link, "collision", name=f"{base}_collision")
        geom_c = ET.SubElement(collision, "geometry")
        mesh_c = ET.SubElement(geom_c, "mesh")
        uri_c = ET.SubElement(mesh_c, "uri")
        uri_c.text = f"model://gazebo_model/meshes/{coll}"

# optional: a simple static inertial (helps some viewers). Not necessary.
# write file with pretty print
def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            indent(e, level+1)
        if not e.tail or not e.tail.strip():
            e.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i

indent(sdf)
tree = ET.ElementTree(sdf)
tree.write(OUT_SDF, encoding="utf-8", xml_declaration=True)
print("Wrote", OUT_SDF)
