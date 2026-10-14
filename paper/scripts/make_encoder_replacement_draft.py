"""Draw an editable replacement for the learned tokenizer's encoder diagram.

This is a separate adaptation draft. It follows ChannelPatchEncoder,
CausalResidualBlock, and ChannelwiseEMGTokenizer, verified against the
released learned tokenizer checkpoint on 2026-09-21. It does not change the manuscript,
existing figure, Affinity document, model, or checkpoint.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
import argparse
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time

import paths

WIDTH, HEIGHT = 1060, 1670
CX = 670
COLORS = dict(temporal="#fbe3c3", pointwise="#d2e7f3", activation="#f5f3c6", norm="#d9ead3")


def build() -> str:
    parts = [f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}"
     viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title description">
<title id="title">Learned EMG encoder: corrected replacement draft</title>
<desc id="description">One normalized EMG channel is embedded with a temporal
convolution from one to 32 features, GELU, and a pointwise projection to 256.
Four pre-normalized causal residual blocks each add their input to their output.
A final LayerNorm precedes vector quantization. The 160 millisecond patch and
320 milliseconds of additional temporal context give a 480 millisecond receptive field.</desc>
<style>
text {{ font-family: Arial, Helvetica, sans-serif; fill: #171717; }}
.group-label {{ font-size: 32px; font-weight: 700; }}
.block-label {{ font-size: 31px; text-anchor: middle; }}
.detail {{ font-size: 22px; text-anchor: middle; }}
.note {{ font-size: 22px; fill: #444444; }}
.container {{ fill: none; stroke: #c6c6c6; stroke-width: 5;
              stroke-dasharray: 23 14; stroke-linecap: round; }}
.wire {{ fill: none; stroke: #303030; stroke-width: 4.5;
         stroke-linejoin: round; }}
</style>
<defs><marker id="arrow" viewBox="0 0 12 12" refX="11" refY="6"
markerWidth="7" markerHeight="7" orient="auto">
<path d="M 1 1 L 11 6 L 1 11 Z" fill="#303030"/>
</marker></defs>''']

    def text(x, y, label, cls="note", anchor=None):
        alignment = f' text-anchor="{anchor}"' if anchor else ""
        parts.append(f'<text x="{x}" y="{y}" class="{cls}"{alignment}>{escape(label)}</text>')

    def arrow(y0, y1):
        parts.append(f'<path class="wire" d="M {CX} {y0} V {y1}" marker-end="url(#arrow)"/>')

    def box(y, h, w, color, label, detail=None, element_id=""):
        x = CX - w / 2
        ident = f' id="{element_id}"' if element_id else ""
        parts.append(f'<g{ident}><rect x="{x}" y="{y}" width="{w}" height="{h}" '
                     f'rx="15" fill="{COLORS[color]}" stroke="#303030" stroke-width="3.5"/>')
        if detail:
            text(CX, y + 37, label, "block-label")
            text(CX, y + 67, detail, "detail")
        else:
            text(CX, y + h / 2 + 11, label, "block-label")
        parts.append('</g>')

    # All labels remain native editable text; groups match editing boundaries.
    parts.append('<g id="input">')
    text(CX, 45, "Normalized EMG", "block-label")
    text(CX, 76, "One electrode | Sample interval: 0.5 ms", "detail")
    arrow(91, 156)
    parts.append('</g>')

    parts.append('<g id="patch-embedding">')
    parts.append('<rect class="container" x="30" y="125" width="1000" height="370" rx="26"/>')
    text(72, 200, "Patch", "group-label")
    text(72, 239, "embedding", "group-label")
    text(72, 318, "160 ms window")
    text(72, 351, "20 ms hop")
    box(165, 86, 565, "temporal", "Temporal Conv1D", "32 filters | 160 ms window | 20 ms hop", "patch-temporal")
    arrow(251, 284)
    box(293, 59, 220, "activation", "GELU", element_id="patch-gelu")
    arrow(352, 385)
    box(394, 86, 490, "pointwise", "Pointwise 1×1", "32 → 256", "patch-pointwise")
    parts.append('</g>')

    # The branch is taken before LayerNorm. The complete dashed region, including
    # its identity path and addition, is repeated with four separate blocks.
    parts.append('<g id="residual-stack">')
    parts.append('<rect class="container" x="30" y="550" width="1000" height="794" rx="26"/>')
    text(142, 837, "4 × residual", "group-label")
    text(142, 878, "block", "group-label")
    text(142, 954, "320 ms", "group-label")
    text(142, 990, "additional history")
    text(142, 1032, "Repeat entire block", "note")
    arrow(480, 626)
    parts.append(f'<circle cx="{CX}" cy="591" r="5.5" fill="#303030"/>')
    parts.append(f'<path class="wire" d="M {CX} 591 H 95 V 1272 H {CX - 39}" marker-end="url(#arrow)"/>')
    box(635, 62, 370, "norm", "LayerNorm", element_id="residual-prenorm")
    arrow(697, 733)
    # Five token positions are the current vector and four preceding vectors:
    # 4 * 20 ms = 80 ms lookback per block, rather than 100 ms lookback.
    box(742, 86, 595, "temporal", "Depthwise causal Conv1D", "256 features | 80 ms lookback | 20 ms hop", "residual-depthwise")
    arrow(828, 864)
    box(873, 86, 490, "pointwise", "Pointwise 1×1", "256 → 1,024", "residual-expand")
    arrow(959, 995)
    box(1004, 59, 220, "activation", "GELU", element_id="residual-gelu")
    arrow(1063, 1099)
    box(1108, 86, 490, "pointwise", "Pointwise 1×1", "1,024 → 256", "residual-project")
    arrow(1194, 1236)
    parts.append(f'<g id="residual-add"><circle cx="{CX}" cy="1272" r="31" fill="#ffffff" '
                 'stroke="#303030" stroke-width="4"/>')
    parts.append(f'<path class="wire" d="M {CX - 16} 1272 H {CX + 16} M {CX} 1256 V 1288"/></g>')
    parts.append('</g>')

    parts.append('<g id="pre-quantizer-normalization">')
    arrow(1303, 1381)
    box(1390, 62, 370, "norm", "LayerNorm", element_id="final-layernorm")
    text(72, 1429, "After all 4 blocks", "note")
    arrow(1452, 1512)
    text(CX, 1550, "256-D features h", "block-label")
    text(CX, 1585, "480 ms effective receptive field", "detail")
    text(CX, 1623, "To vector quantizer → token ID", "detail")
    parts.append('</g></svg>')
    return '\n'.join(parts) + '\n'


def render(svg: Path, width: int = WIDTH, height: int = HEIGHT) -> None:
    chrome = Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
    if not chrome.exists() or not shutil.which('pdftoppm'):
        raise RuntimeError('Rendering requires Chrome and pdftoppm; the editable SVG is already written.')
    with tempfile.TemporaryDirectory(prefix='encoder-replacement-render-') as temp:
        directory = Path(temp)
        shutil.copy2(svg, directory / svg.name)
        html = directory / 'figure.html'
        html.write_text(f'''<!doctype html><html><head><meta charset="utf-8"><style>
@page {{size: {width}px {height}px; margin: 0;}}
html,body {{margin:0; padding:0; background:white;}}
img {{display:block; width:{width}px; height:{height}px;}}
</style></head><body><img src="{svg.name}"></body></html>''')
        rendered = directory / 'rendered.pdf'
        # Some Chrome builds stay alive after printing. Use successful PDF
        # validation as completion, then stop only this isolated process group.
        with (directory/'chrome.log').open('w') as log:
            process = subprocess.Popen([str(chrome), '--headless=new', '--disable-gpu',
                '--disable-background-networking', '--no-first-run',
                f'--user-data-dir={directory / "chrome-profile"}', '--no-pdf-header-footer',
                f'--print-to-pdf={rendered}', html.as_uri()],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            valid = False
            try:
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if rendered.exists() and rendered.stat().st_size:
                        check = subprocess.run(['pdfinfo', str(rendered)], capture_output=True, text=True)
                        if check.returncode == 0:
                            valid = True
                            break
                    if process.poll() is not None:
                        break
                    time.sleep(.15)
            finally:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=3)
        if not valid:
            raise RuntimeError((directory/'chrome.log').read_text()[-2000:])
        shutil.copy2(rendered, svg.with_suffix('.pdf'))
        subprocess.run(['pdftoppm', '-png', '-r', '96', '-singlefile',
            str(svg.with_suffix('.pdf')), str(svg.with_suffix(''))], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=paths.OUTPUT/'figures/encoder-replacement-draft.svg')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    svg = args.output.resolve()
    svg.parent.mkdir(parents=True, exist_ok=True)
    svg.write_text(build())
    if args.render:
        render(svg)
    print(json.dumps(dict(svg=str(svg), width=WIDTH, height=HEIGHT, rendered=args.render), indent=2))


if __name__ == '__main__':
    main()
