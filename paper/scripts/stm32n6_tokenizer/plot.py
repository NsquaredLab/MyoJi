"""Plot only physical-board measurements listed in a frozen run manifest."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
import seaborn as sns
from build import ROOT, OUT, sha
sys.path.insert(0, str(ROOT/'scripts'))
from paths import DATA  # noqa: E402

# The frozen board measurements behind Fig. 6 also ship in data/firmware/, laid
# out like results/, so the figure can be redrawn without the board. A local
# collection in results/ takes precedence.
FIRMWARE = DATA/'firmware'


def measured(path):
    """`path` (under results/) if it exists, else its shipped copy under data/firmware/."""
    path = Path(path)
    return path if path.exists() else FIRMWARE/path.relative_to(ROOT/'results')

STYLE = ROOT/'scripts/simpetru.mplstyle'  # the paper's figure style (see scripts/figstyle.py)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=measured(OUT/'measurement_manifest.json'))
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    grouped={}; rejected=[]
    for item in manifest['runs']:
        folder=measured(ROOT/item['path'])
        assert sha(folder/'receipt.json')==item['receipt_sha256']
        assert sha(folder/'frames.csv')==item['frames_sha256']
        receipt=json.loads((folder/'receipt.json').read_text())
        config=receipt['configuration'];device=receipt['device']
        assert receipt['hardware'] and device['hardware']==1 and device['status']==3
        assert device['cpu_hz']==600000000 and device['cache_enabled']==1
        assert config['window_ms']==160 and config['update_ms']==20
        assert config['frames']==manifest['frames_per_run']
        assert config['warmup']==manifest['warmup_frames']
        assert config['seed']==manifest['seed']
        assert config['elf_sha256']==manifest['elf_sha256']
        rows=list(csv.DictReader((folder/'frames.csv').open()))
        assert len(rows)==config['frames']
        if not receipt['output_verified']:
            rejected.append({'run':item['path'],'route':config['route'],
                             'channels':config['channels'],'reason':'Token IDs/checksums differ from host'})
            continue
        for row in rows:
            assert row['hardware']=='True' and row['output_verified']=='True'
            assert int(row['cpu_hz'])==device['cpu_hz']
        values=np.asarray([int(row['total_cycles'])/device['cpu_hz']*1000 for row in rows])
        assert np.isfinite(values).all() and (values>0).all()
        grouped.setdefault((config['route'],config['channels']),[]).append((values,item['path']))
    summary=[]
    for (route,n),items in sorted(grouped.items()):
        values=np.concatenate([item[0] for item in items]);rate=n*1000/values
        summary.append({'route':route,'channels':n,'frames':len(values),
            'runs':len(items),'median_ms':float(np.median(values)),
            'min_ms':float(values.min()),'max_ms':float(values.max()),
            'median_tokens_per_s':float(np.median(rate)),
            'min_tokens_per_s':float(rate.min()),'max_tokens_per_s':float(rate.max()),
            'all_frames_within_20ms':bool((values<=20).all()),
            'sources':[item[1] for item in items]})
    if not summary: raise RuntimeError('No verified physical-board measurements')
    dest=ROOT/'output/figures/stm32n6_tokenizer';dest.mkdir(parents=True,exist_ok=True)
    largest={route:max([r['channels'] for r in summary
        if r['route']==route and r['all_frames_within_20ms']],default=0)
        for route in ['spar','learned']}
    report={'manifest_sha256':sha(args.manifest),'summary':summary,
            'excluded_runs':rejected,'largest_tested_verified_within_20ms':largest,
            'timing':'Median and min–max across sequential synthetic frames; not a confidence interval.'}
    (dest/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    with (dest/'summary.csv').open('w',newline='') as f:
        fields=[k for k in summary[0] if k!='sources']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in summary)

    plt.style.use(STYLE)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],
        'font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
        'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none','text.usetex':False,
        'savefig.bbox':None})
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.95))
    fig.subplots_adjust(left=.135,right=.985,bottom=.20,top=.82,wspace=.40)
    settings={'spar':('#d80032','o','SPAR + VQ'),
              'learned':('#5c618e','s','Learned encoder + VQ')}
    handles=[]
    for route,(color,marker,label) in settings.items():
        data=[r for r in summary if r['route']==route]
        if not data: continue
        n=np.array([r['channels'] for r in data])
        failed={r['channels'] for r in rejected if r['route']==route}
        for ax,key,lower,upper in [(axes[0],'median_ms','min_ms','max_ms'),
                                   (axes[1],'median_tokens_per_s','min_tokens_per_s','max_tokens_per_s')]:
            y=np.array([r[key] for r in data])
            # Do not visually bridge configurations that failed validation.
            for i in range(1,len(n)):
                if not any(n[i-1]<bad<n[i] for bad in failed):
                    ax.plot(n[i-1:i+1],y[i-1:i+1],color=color,lw=1,alpha=.8)
            ax.errorbar(n,y,yerr=[y-np.array([r[lower] for r in data]),
                        np.array([r[upper] for r in data])-y],fmt=marker,
                        color=color,markersize=4.2,elinewidth=.8,capsize=2,
                        markeredgecolor='white',markeredgewidth=.4,zorder=3)
        handles.append(Line2D([],[],color=color,marker=marker,ms=4,lw=1,label=label))
    maximum=max(r['channels'] for r in summary)
    ticks=[n for n in [1,4,16,64,320,1024] if n<=maximum]
    decimal=FuncFormatter(lambda value,_:f'{value:,.0f}' if value>=1 else f'{value:g}')
    for index,ax in enumerate(axes):
        ax.set_xscale('log');ax.set_yscale('log')
        ax.set_xlim(.85,maximum*1.18)
        ax.xaxis.set_major_locator(FixedLocator(ticks));ax.xaxis.set_major_formatter(decimal)
        ax.xaxis.set_minor_locator(NullLocator());ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(decimal)
        ax.set_xlabel('EMG channels')
        ax.grid(axis='both',color='#e2e2e2',linewidth=.55)
        ax.text(-.18,1.04,chr(65+index),transform=ax.transAxes,fontweight='bold',fontsize=11)
        ax.tick_params(which='both',top=False,right=False,direction='out')
    axes[0].set_ylabel('Compute time per update (ms)')
    axes[0].axhline(20,color='#6d6f72',ls='--',lw=.85,zorder=1)
    axes[0].annotate('20 ms deadline',(.97,20),xycoords=('axes fraction','data'),
        xytext=(0,4),textcoords='offset points',ha='right',va='bottom',fontsize=8,color='#6d6f72')
    if largest['spar']:
        selected=next(r for r in summary if r['route']=='spar' and r['channels']==largest['spar'])
        axes[0].annotate(f"{selected['channels']} channels\n{selected['median_ms']:.1f} ms",
            (selected['channels'],selected['median_ms']),xytext=(-12,-31),
            textcoords='offset points',ha='center',va='top',fontsize=8,color=settings['spar'][0],
            arrowprops={'arrowstyle':'-','color':settings['spar'][0],'lw':.6,'shrinkA':3,'shrinkB':5})
    vals=[r['min_ms'] for r in summary]+[r['max_ms'] for r in summary]
    top=10**np.ceil(np.log10(max(vals)*1.08));bottom=10**np.floor(np.log10(min(vals)*.8))
    axes[0].set_ylim(bottom,top)
    axes[0].set_yticks([v for v in [.1,1,10,100,1000,10000,100000] if bottom<=v<=top])
    axes[1].set_ylabel('Token IDs per second')
    # The line is the required aggregate rate, not measured BLE throughput.
    xs=np.geomspace(1,maximum,200)
    axes[1].plot(xs,50*xs,color='#6d6f72',ls='--',lw=.85,zorder=1)
    handles.append(Line2D([],[],color='#6d6f72',ls='--',lw=.85,label='50 Hz requirement'))
    rates=[r['min_tokens_per_s'] for r in summary]
    bottom=10**np.floor(np.log10(min(rates)*.8));top=10**np.ceil(np.log10(max(50*maximum,max(r['max_tokens_per_s'] for r in summary))*1.08))
    axes[1].set_ylim(bottom,top)
    axes[1].set_yticks([v for v in [1,10,100,1000,10000,100000] if bottom<=v<=top])
    for ax in axes: sns.despine(ax=ax,trim=True,offset=3)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,1),frameon=False,ncol=3)
    for ext in ['pdf','svg','png']:
        fig.savefig(dest/f'stm32n6-tokenizer-scaling.{ext}',dpi=300,bbox_inches=None,transparent=False)
    plt.close(fig)
    caption=('Physical-board throughput of the frozen tokenizers on a NUCLEO-N657X0-Q '
        '(Cortex-M55 CPU, FP32, 600 MHz, instruction and data caches enabled; no NPU). '
        'Synthetic 2 kHz input is processed in 160 ms windows with 20 ms shifts. '
        'Timing includes window updates, normalization, the complete frontend, exhaustive '
        '1,024-code assignment and 10-bit packing; sample generation, one-off calibration, '
        'ADC and BLE transfers are excluded. Points show medians; whiskers show the observed '
        f'minimum and maximum of {manifest["frames_per_run"]} sequential frames per run, '
        f'on one board, after {manifest["warmup_frames"]} warmup frames per configuration. '
        'These ranges are not confidence intervals. '
        'Only runs with exact host token-ID/checksum agreement are plotted; failed '
        'configurations are retained in the accompanying validation report. '
        + ('SPAR channel counts '+', '.join(str(n) for n in sorted({r['channels'] for r in rejected if r['route']=='spar'}))+
           ' were attempted but failed exact token validation. ' if any(r['route']=='spar' for r in rejected) else '')+
        'Connecting lines '
        'guide the eye and do not establish untested channel capacities. A: compute time '
        'per update; the dashed line is the 20 ms deadline. B: aggregate token production '
        'rate, computed as channels divided by update time, with one ID per channel '
        'per update; the dashed line is 50 times the channel count required for 50 Hz updates. '
        'This compute-limited rate is not measured acquisition or radio throughput. '
        'The learned route reads its 9.57 MB frozen weights from 50 MHz OPI DTR external NOR; SPAR constants are '
        'internal. These measurements characterize this CPU port, not the peak capability '
        'of the STM32N6 accelerator.')
    (dest/'caption.txt').write_text(caption+'\n')
    (dest/'include.tex').write_text('\\begin{figure*}[t]\n\\centering\n'
        '\\includegraphics[width=\\textwidth]{figures/stm32n6-tokenizer-scaling.pdf}\n'
        '\\caption{'+caption.replace('%','\\%')+'}\n\\label{fig:stm32n6-throughput}\n\\end{figure*}\n')
    print(dest/'stm32n6-tokenizer-scaling.pdf')
    print('Largest tested verified configuration within deadline:',largest)
    print('Rejected runs:',len(rejected))


if __name__=='__main__': main()
