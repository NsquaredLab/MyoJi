"""Build a manually laid out, source-linked vector mechanism figure.

TikZ is the editable architecture source. Actual validation descriptors,
centroids, tokens and one validation-only pose output supply numerical insets.
The schematic waveform is not a recorded trace. No fitting or test scoring.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch

from paper.common.tokenizer_diagnostic_style import apply, FEATURE, TOKEN, INK
from paper.common import paths

ROOT = Path(__file__).resolve().parents[1]
OUT = paths.OUTPUT/'figures/pose_paper'
BUILD = paths.OUTPUT/'build/tokenizer-mechanism'
SOURCE = ROOT/'results/dpose_spar_simple_full_v1/subject_01/figure-video-inputs.npz'
CKPT = ROOT/'results/dpose_spar_simple_full_v1/subject_01/fits/tokens_260911/best.pt'
TIKZ = paths.DATA/'figure_inputs/tokenizer_mechanism.tex'
EXAMPLE = dict(participant=1, validation_window=0, frame=49, channel=27)


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    """Source-hash key: relative to paper/, else to the repository (library files)."""
    return str(p.relative_to(ROOT if p.is_relative_to(ROOT) else paths.REPO))


def command(args):
    r=subprocess.run([str(v) for v in args],cwd=BUILD,capture_output=True,text=True)
    if r.returncode:
        (BUILD/'command-error.log').write_text(r.stdout+'\n'+r.stderr)
        raise RuntimeError('See '+str(BUILD/'command-error.log')+'\n'+(r.stdout+r.stderr)[-1800:])
    return r.stdout+r.stderr


def rgb(c): return ','.join(f'{v:.7f}' for v in c[:3])


def build():
    OUT.mkdir(parents=True,exist_ok=True);BUILD.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    apply()
    data=np.load(SOURCE,allow_pickle=False)
    w,t,c=EXAMPLE['validation_window'],EXAMPLE['frame'],EXAMPLE['channel']
    h=data['val_continuous'][w,c,t].copy();book=data['codebook']
    distance=np.sum((book.astype(np.float64)-h.astype(np.float64))**2,axis=1)
    closest=np.argsort(distance,kind='stable')[:3]
    token=int(data['val_tokens'][w,c,t])
    assert token==int(closest[0])==239
    assert h.shape==(18,) and book.shape==(1024,18) and data['mask'][c].all()
    history=data['val_tokens'][w,25:29,45:50].copy()
    assert int(history[2,-1])==token
    # Fixed symmetric color range shared by descriptor and codebook insets.
    values=np.r_[h,book[closest].ravel()]
    limit=4.0
    assert np.max(np.abs(values))<limit
    cmap=LinearSegmentedColormap.from_list('descriptor',['#F2F2F6','#ABADC5','#2B2E4C'])
    text=[]
    def color(name,value):
        text.append(rf'\definecolor{{{name}}}{{rgb}}{{{rgb(cmap((float(value)+limit)/(2*limit)))}}}')
    for d,v in enumerate(h):color(f'h{d}',v)
    for k in range(3):
        for d,v in enumerate(book[closest[k]]):color(f'e{k}c{d}',v)
    for i,v in enumerate(np.linspace(-limit,limit,32)):color(f'key{i}',v)
    def macro(name,body):text.extend([rf'\newcommand{{\{name}}}{{',*body,'}'])
    macro('ExampleToken',[str(token)])
    macro('DescriptorStrip',[
        rf'\filldraw[fill=h{d},draw=white,line width=.2pt] ({d*.182:.4f},0) rectangle ({(d+1)*.182:.4f},.34);'
        for d in range(18)])
    macro('DescriptorMatrix',[
        rf'\filldraw[fill=h{d},draw=white,line width=.2pt] ({d%3*.22:.4f},{(5-d//3)*.185:.4f}) rectangle ({(d%3+1)*.22:.4f},{(6-d//3)*.185:.4f});'
        for d in range(18)])
    rows=[]
    for k,idx in enumerate(closest):
        yy=-k*.56
        rows += [rf'\node[anchor=east,text={"token" if k==0 else "gray"}] at (-.18,{yy+.17:.4f}) {{$e_{{{idx}}}$}};']
        for d in range(18):
            rows += [rf'\filldraw[fill=e{k}c{d},draw=white,line width=.2pt] ({d*.182:.4f},{yy:.4f}) rectangle ({(d+1)*.182:.4f},{yy+.34:.4f});']
        if k==0:rows += [rf'\draw[token,line width=1.05pt] (-.04,{yy-.04:.4f}) rectangle ({18*.182+.04:.4f},{yy+.38:.4f});']
    macro('CodebookRows',rows)
    key=[rf'\fill[key{i}] ({i*2.36/32:.4f},0) rectangle ({(i+1)*2.36/32:.4f},.13);' for i in range(32)]
    key += [rf'\node[note,anchor=north] at ({x},-.03) {{{v}}};' for x,v in [(0,'$-4$'),(1.18,'$0$'),(2.36,'$4$')]]
    macro('ValueKey',key)
    bits=format(token,'010b')
    macro('TokenBits',[
        rf'\node[draw=token!55,fill=token!{12 if bit=="1" else 2},text=token,minimum width=.43cm,minimum height=.37cm,inner sep=0pt] at ({i*.45:.3f},0) {{{bit}}};'
        for i,bit in enumerate(bits)])
    tiles=[]
    for r in range(4):
        for col in range(5):
            selected=(r==2 and col==4)
            tiles += [rf'\node[draw=white,line width=.4pt,fill={"token" if selected else "token!7"},text={"white" if selected else "ink"},font=\fontsize{{8}}{{8.5}}\selectfont,minimum width=.58cm,minimum height=.28cm,inner sep=0pt] at ({col*.59:.3f},{(3-r)*.29:.3f}) {{{int(history[r,col])}}};']
    # Ellipsis identifies a crop, rather than pretending four rows are all channels.
    tiles += [r'\node[note] at (2.97,.44) {$\vdots$};']
    macro('TokenHistory',tiles)
    (BUILD/'example.tex').write_text('\n'.join(text)+'\n')

    # A fixed validation-only forward pass provides the actual output hand.
    from myoji._vendor.raulnet_v20 import RaulNetV20
    checkpoint=torch.load(CKPT,map_location='cpu',weights_only=True)
    model=RaulNetV20(torch.from_numpy(book))
    model.load_state_dict({k.removeprefix('decoder.'):v for k,v in checkpoint['model'].items()
                           if k.startswith('decoder.')},strict=True)
    model.eval().requires_grad_(False)
    mask=checkpoint['model']['mask']
    np.testing.assert_array_equal(mask.numpy(),data['mask'])
    np.testing.assert_array_equal(model.codebook.numpy(),book)
    with torch.inference_mode():
        # Exact SparPose.forward operation, without importing the HDF5 trainer.
        ids=torch.from_numpy(data['val_tokens'][w:w+1].astype(np.int64))
        vectors=torch.nn.functional.embedding(ids,model.codebook)*mask[None,:,None,:]
        output=model.forward_embeddings(vectors)
        output=(output*checkpoint['scale']+checkpoint['mean']).numpy()[0,t].reshape(20,3)
    points=np.vstack([np.zeros((1,3)),output])
    assert np.isfinite(points).all()
    # Palm-aligned orthographic drawing; this rigid display transform does not
    # alter predictions or select examples by their errors.
    up=points[[5,9,13,17]].mean(0)-points[0]
    up=up/np.linalg.norm(up)
    right=points[17]-points[5]
    right=right-up*np.dot(right,up)
    right=right/np.linalg.norm(right)
    xy=points@np.stack([right,up],axis=1)
    fig,ax=plt.subplots(figsize=(1.25,1.13));fig.subplots_adjust(0,0,1,1)
    for start in [1,5,9,13,17]:
        idx=[0,*range(start,start+4)]
        ax.plot(xy[idx,0],xy[idx,1],color=INK,lw=1.3,zorder=1)
    ax.scatter(xy[1:,0],xy[1:,1],s=8,color=TOKEN,zorder=2,edgecolor='white',linewidth=.3)
    ax.scatter(*xy[0],s=12,color=INK,zorder=3)
    ax.set_aspect('equal');ax.margins(.08);ax.axis('off')
    fig.savefig(BUILD/'pose.pdf',bbox_inches='tight',pad_inches=.005)
    plt.close(fig)
    np.savez_compressed(BUILD/'example-data.npz',descriptor=h,codebook_rows=book[closest],closest=closest,
                        distances=distance[closest],token_history=history,pose_mm=points)
    array_output=ROOT/'output/figures/tokenizer-mechanism-data.npz'
    array_output.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(BUILD/'example-data.npz',array_output)
    shutil.copy2(TIKZ,BUILD/'mechanism.tex')
    log=command(['pdflatex','-interaction=nonstopmode','-halt-on-error','-no-shell-escape','mechanism.tex'])
    (BUILD/'build.log').write_text(log)
    assert 'Overfull' not in log and 'undefined' not in log
    pdf=BUILD/'mechanism.pdf'
    command(['pdftocairo','-svg',pdf,BUILD/'mechanism.svg'])
    command(['pdftoppm','-singlefile','-r','300','-png',pdf,BUILD/'mechanism'])
    for ext in ['pdf','svg','png']:
        shutil.copy2(BUILD/f'mechanism.{ext}',OUT/f'interface.{ext}')
        target=ROOT/('output/pdf' if ext=='pdf' else 'output/figures')/f'tokenizer-mechanism.{ext}'
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BUILD/f'mechanism.{ext}',target)
    sources={rel(p):sha(p) for p in [SOURCE,CKPT,TIKZ,Path(__file__),
        paths.REPO/'src/myoji/codebook.py',paths.REPO/'src/myoji/spar.py',
        ROOT/'hdemg/run_dpose_spar_simple.py',paths.REPO/'src/myoji/decoders.py',
        paths.REPO/'src/myoji/_vendor/raulnet_v20.py',
        paths.REPO/'src/myoji/learned/encoder.py',
        paths.REPO/'src/myoji/learned/quantizer.py',paths.REPO/'src/myoji/learned/config.py']}
    receipt=dict(complete=True,schematic=True,manual_vector_layout=True,
        scope='Learned and SPAR frontends, frozen nearest-code assignment, receiver lookup and supervised pose decoder',
        example=EXAMPLE,token=token,binary_value_msb_first=bits,nearest_ids=closest.tolist(),
        squared_distances_float64=distance[closest].tolist(),standardized_component_color_limits=[-limit,limit],
        color_clipping=False,full_descriptor_dim=18,codebook_shape=list(book.shape),
        token_history_channels=list(range(25,29)),token_history_frames=list(range(45,50)),
        pose_kind='Actual fixed-seed SPAR-token validation forward pass; palm-aligned orthographic drawing, not an accuracy comparison',
        decoder_seed=260911,pose_labels_used_for_selection=False,test_data_used=False,new_training=False,
        waveform_is_schematic=True,learned_feature_strip_is_schematic=True,
        source_sha256=sources,pdf_sha256=sha(OUT/'interface.pdf'),
        example_arrays=str(array_output.relative_to(ROOT)),example_arrays_sha256=sha(array_output),
        figure_width_inches=7.16,figure_height_inches=16.2/2.54)
    (OUT/'interface.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (ROOT/'output/figures/tokenizer-mechanism.json').write_text(json.dumps(receipt,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/'output/figures/tokenizer-mechanism-source.zip','w',zipfile.ZIP_DEFLATED) as bundle:
        for name in ('mechanism.tex','example.tex','pose.pdf'):
            bundle.write(BUILD/name,name)
        bundle.write(array_output,'example-data.npz')
        bundle.write(OUT/'interface.json','provenance.json')
        bundle.writestr('README.md',
            '# Tokenizer mechanism figure\n\n'
            'Editable vector layout: mechanism.tex. Compile with pdfLaTeX (TikZ, newtxtext, newtxmath).\n'
            'example.tex contains the source-linked numerical insets; pose.pdf is a vector validation prediction.\n'
            'The waveform and learned-feature strip are schematic. No new fitting or test evaluation.\n'
            'Run `uv run python -m paper.figures.build_tokenizer_mechanism` to regenerate from the saved arrays and checkpoint.\n'
            'The standalone PDF is 7.16 inches wide. The paper caption supplies full example provenance.\n')
    return receipt


if __name__=='__main__':
    print(json.dumps({k:v for k,v in build().items() if k in ['complete','token','nearest_ids','pdf_sha256']}))
