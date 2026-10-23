"""Export the existing feature plots as simple, editable Affinity import files.

Keeps the plotted coordinates and values. SVG uses individual circles instead
of referenced markers; PDF uses outlined text and contains no embedded fonts.
Original exports and data are preserved. This is a compatibility workaround,
not a claim that a specific PDF construct caused the reported import crash.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath, TextToPath
from matplotlib.transforms import Affine2D

from paper.figures.make_encoder_replacement_draft import render
from paper.common import paths

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = paths.OUTPUT/'figures/encoder_examples'
NS = 'http://www.w3.org/2000/svg'
XLINK = 'http://www.w3.org/1999/xlink'
ET.register_namespace('', NS)


def style_attributes(style):
    return dict(item.strip().split(':', 1) for item in style.split(';') if ':' in item)


def simple_svg(source):
    root = ET.parse(source).getroot()
    ids = {e.attrib['id']: e for e in root.iter() if 'id' in e.attrib}
    collection = ids.get('PathCollection_1')
    coordinates = np.empty((0,2))
    if collection is not None:
        markers = collection.findall(f'.//{{{NS}}}use')
        assert len(markers) == 256
        coordinates = np.asarray([(float(e.get('x')), float(e.get('y'))) for e in markers])
        # Current scatter markers are radius 1 pt, as defined by the source path.
        marker_shape = ids[markers[0].get(f'{{{XLINK}}}href')[1:]]
        assert marker_shape.get('d').strip().startswith('M 0 1')
        collection.clear()
        collection.set('id', 'feature-dots')
        for index, (x, y) in enumerate(coordinates, 1):
            ET.SubElement(collection, f'{{{NS}}}circle',
                {'id':f'feature-{index:03d}', 'cx':str(x), 'cy':str(y), 'r':'1',
                 'fill':'#5c618e', 'stroke':'none'})
    # Expand the remaining reused objects (axis tick paths) locally.
    for parent in root.iter():
        for child in list(parent):
            if child.tag == f'{{{NS}}}use':
                replacement = deepcopy(ids[child.get(f'{{{XLINK}}}href')[1:]])
                replacement.attrib.pop('id', None)
                replacement.set('transform', f"translate({child.get('x','0')} {child.get('y','0')})")
                if child.get('style'):
                    replacement.set('style', child.get('style'))
                parent.insert(list(parent).index(child), replacement)
                parent.remove(child)
    for parent in root.iter():
        for child in list(parent):
            if child.tag in {f'{{{NS}}}defs', f'{{{NS}}}metadata'}:
                parent.remove(child)
    for element in root.iter():
        element.attrib.pop('clip-path', None)
        style = style_attributes(element.attrib.pop('style', ''))
        for key, value in style.items():
            element.set(key.strip(), value.strip())
        if element.tag == f'{{{NS}}}text':
            element.set('font-family', 'Arial')
    assert not root.findall(f'.//{{{NS}}}use')
    assert not root.findall(f'.//{{{NS}}}clipPath')
    assert not root.findall(f'.//{{{NS}}}image')
    return root, coordinates


def outline_text(root):
    root = deepcopy(root)
    for parent in root.iter():
        for element in list(parent):
            if element.tag != f'{{{NS}}}text':
                continue
            label = ''.join(element.itertext())
            size = float(element.get('font-size').removesuffix('px'))
            font = FontProperties(family='Arial', size=size)
            advance, _, _ = TextToPath().get_text_width_height_descent(label, font, ismath=False)
            anchor = element.get('text-anchor', 'start')
            adjustment = {'start':0, 'middle':advance/2, 'end':advance}[anchor]
            x, y = float(element.get('x', '0')), float(element.get('y', '0'))
            transform = Affine2D().scale(1, -1).translate(x-adjustment, y)
            rotation = re.fullmatch(r'rotate\(([-\d.]+) ([-\d.]+) ([-\d.]+)\)', element.get('transform',''))
            translated_rotation = re.fullmatch(
                r'translate\(([-\d.]+) ([-\d.]+)\) rotate\(([-\d.]+)\)', element.get('transform',''))
            translation = re.fullmatch(r'translate\(([-\d.]+) ([-\d.]+)\)', element.get('transform',''))
            if rotation:
                angle, cx, cy = map(float, rotation.groups())
                transform = transform.rotate_deg_around(cx, cy, angle)
            elif translated_rotation:
                tx, ty, angle = map(float, translated_rotation.groups())
                transform = transform.rotate_deg(angle).translate(tx, ty)
            elif translation:
                transform = transform.translate(*map(float, translation.groups()))
            elif element.get('transform'):
                raise ValueError(element.get('transform'))
            path = TextPath((0,0), label, prop=font, size=size).transformed(transform)
            chunks = []
            commands = {MplPath.MOVETO:'M', MplPath.LINETO:'L', MplPath.CURVE3:'Q', MplPath.CURVE4:'C'}
            for values, code in path.iter_segments(curves=True, simplify=False):
                if code == MplPath.CLOSEPOLY:
                    chunks.append('Z')
                else:
                    chunks.append(commands[code]+' '+' '.join(f'{v:.6f}' for v in values))
            replacement = ET.Element(f'{{{NS}}}path', {'d':' '.join(chunks), 'fill':'#000000', 'stroke':'none'})
            ET.SubElement(replacement, f'{{{NS}}}title').text = label
            parent.insert(list(parent).index(element), replacement)
            parent.remove(element)
    assert not root.findall(f'.//{{{NS}}}text')
    return root


def write(root, path):
    path.write_bytes(ET.tostring(root, encoding='utf-8', xml_declaration=True))


def main():
    reports = []
    for stem in ['normalized-emg-channel', 'normalized-emg-inset',
                 'prequantization-vector', 'prequantization-vector-inset']:
        source = DIRECTORY/f'{stem}.svg'
        tree, expected = simple_svg(source)
        target = DIRECTORY/f'{stem}-affinity.svg'
        write(tree, target)
        outlined = DIRECTORY/f'{stem}-affinity-outlined.svg'
        write(outline_text(tree), outlined)
        width, height = map(float, tree.get('viewBox').split()[2:])
        # SVG coordinates are in points; Chrome's physical page dimensions use px.
        render(outlined, width=width*96/72, height=height*96/72)
        for ext in ['.pdf', '.png']:
            outlined.with_suffix(ext).replace(target.with_suffix(ext))
        current = np.asarray([(float(e.get('cx')),float(e.get('cy')))
            for e in ET.parse(target).getroot().findall(f'.//{{{NS}}}circle')]).reshape(-1,2)
        np.testing.assert_array_equal(current, expected)
        reports.append(dict(source=str(source.relative_to(ROOT)), circles=len(current),
            coordinate_agreement='exact to existing SVG',
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            svg=str(target.relative_to(ROOT)), pdf=str(target.with_suffix('.pdf').relative_to(ROOT)),
            pdf_text='outlined vector paths', svg_text='editable Arial text'))
    (DIRECTORY/'affinity-export.json').write_text(json.dumps(reports, indent=2)+'\n')
    (DIRECTORY/'AFFINITY.md').write_text(
        '# Affinity import copies\n\n'
        'Use `normalized-emg-channel-affinity.svg` for the matching normalized EMG plot, '
        'or `normalized-emg-inset-affinity.svg` for its axis-free waveform. '
        'Use `prequantization-vector-affinity.svg` for editable labels and 256 separate circle shapes. '
        '`prequantization-vector-inset-affinity.svg` is the version without axes. '
        'The matching PDFs contain vector shapes with text converted to outlines (no embedded fonts). '
        'The `-affinity-outlined.svg` versions also provide those text outlines as SVG.\n\n'
        'Waveform paths and all 256 feature positions, values, colours and axis limits match the current source plots. '
        'The two labelled plots are 1.05 × 1.4 inches, with time/feature index running from top to bottom; '
        'both retain matching trimmed spines and 5-point outward offsets. '
        'Original files and data are unchanged. This simplifies import structures; it does not establish '
        'the precise cause of the reported Affinity crash.\n\n'
        'Reproduce: `uv run python -m paper.figures.export_encoder_examples_affinity`.\n'
    )
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
