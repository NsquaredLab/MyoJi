"""Compare frozen, validated physical-board CPU runs; never plot estimates."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np

from paper.firmware.stm32n6_optimized.build import ROOT, OUT, sha
from paper.common.paths import DATA  # noqa: E402

# The frozen board measurements behind Fig. 6 also ship in data/firmware/, laid
# out like results/, so the figure can be redrawn without the board. A local
# collection in results/ takes precedence.
FIRMWARE = DATA/'firmware'


def measured(path):
    """`path` (under results/) if it exists, else its shipped copy under data/firmware/."""
    path = Path(path)
    return path if path.exists() else FIRMWARE/path.relative_to(ROOT/'results')

STYLE = ROOT/'common/simpetru.mplstyle'  # the paper's figure style (see common/figstyle.py)


def read_manifest(path, implementation):
    manifest = json.loads(path.read_text())
    rows, failures = [], []
    for entry in manifest['runs']:
        folder = measured(ROOT / entry['path'])
        for name, field in [('receipt.json', 'receipt_sha256'), ('frames.csv', 'frames_sha256')]:
            assert sha(folder / name) == entry[field]
        receipt = json.loads((folder / 'receipt.json').read_text())
        c, d = receipt['configuration'], receipt['device']
        assert receipt['hardware'] and d['hardware'] == 1 and d['status'] == 3
        assert d['cpu_hz'] == 600000000 and d['cache_enabled'] == 1
        assert c['window_ms'] == 160 and c['update_ms'] == 20
        assert c['frames'] == manifest['frames_per_run'] == 16
        assert c['warmup'] == manifest['warmup_frames'] == 16
        assert c['seed'] == manifest['seed'] == 260923
        assert c['elf_sha256'] == manifest['elf_sha256']
        verified = bool(receipt['output_verified'])
        # Timing is measured the same way whether or not every token matched the
        # host, so the configuration is plotted either way and the agreement is
        # recorded per configuration. See the caption.
        validation = json.loads((folder / 'output_validation.json').read_text())
        expected, actual = validation['expected'], validation['actual']
        agreeing = sum(x == y for x, y in zip(expected['checksums'], actual['checksums']))
        if not verified:
            failures.append(dict(implementation=implementation, route=c['route'],
                                 channels=c['channels'], frames_agreeing=agreeing,
                                 source=entry['path']))
        frames = list(csv.DictReader((folder / 'frames.csv').open()))
        assert len(frames) == 16
        assert all(r['hardware'] == 'True' for r in frames)
        ms = np.array([int(r['total_cycles']) / d['cpu_hz'] * 1000 for r in frames])
        rate = c['channels'] * 1000 / ms
        row = dict(implementation=implementation, route=c['route'], channels=c['channels'],
                   tokens_match_host=verified, frames_agreeing_with_host=agreeing,
                   median_ms=float(np.median(ms)), min_ms=float(min(ms)), max_ms=float(max(ms)),
                   median_tokens_per_s=float(np.median(rate)),
                   min_tokens_per_s=float(min(rate)), max_tokens_per_s=float(max(rate)),
                   all_frames_within_20ms=bool(max(ms) <= 20),
                   frames=16, source=entry['path'],
                   setup_ms=(receipt['setup_cycles']/d['cpu_hz']*1000
                             if 'setup_cycles' in receipt else None),
                   input_bytes=d['input_bytes'], state_bytes=d['state_bytes'], model_bytes=d['model_bytes'])
        for key in ['preprocessing', 'frontend', 'quantizer', 'packing']:
            row[key + '_median_ms'] = float(np.median([int(f[key+'_cycles']) / d['cpu_hz']*1000 for f in frames]))
        rows.append(row)
    return rows, failures, manifest


def main():
    paths = {'original': measured(ROOT/'results/stm32n6_tokenizer_v1/measurement_manifest.json'),
             'optimized': measured(OUT/'measurement_manifest.json')}
    summary, failures = [], []
    manifests = {}
    for name, path in paths.items():
        rows, rejected, manifest = read_manifest(path, name)
        summary.extend(rows); failures.extend(rejected); manifests[name] = manifest
    # The bit-exact build is a separate frozen manifest; read its limit, never hardcode it.
    exact_rows, _, _ = read_manifest(measured(OUT/'measurement_manifest_exact.json'), 'exact')
    exact_spar = [r for r in exact_rows if r['route'] == 'spar']
    largest_exact = max(r['channels'] for r in exact_spar
                        if r['all_frames_within_20ms'] and r['tokens_match_host'])
    largest_exact_timing = max(r['channels'] for r in exact_spar if r['all_frames_within_20ms'])
    largest = {name: {route: max((r['channels'] for r in summary
        if r['implementation'] == name and r['route'] == route
        and r['all_frames_within_20ms'] and r['tokens_match_host']), default=0)
        for route in ['spar', 'learned']} for name in paths}
    speedups = []
    for row in summary:
        if row['implementation'] != 'optimized':
            continue
        baseline = next((r for r in summary if r['implementation'] == 'original' and
                         r['route'] == row['route'] and r['channels'] == row['channels']), None)
        if baseline:
            speedups.append(dict(route=row['route'], channels=row['channels'],
                original_ms=baseline['median_ms'], optimized_ms=row['median_ms'],
                speedup=baseline['median_ms']/row['median_ms']))
    dest = ROOT/'output/figures/stm32n6_optimized'
    dest.mkdir(parents=True, exist_ok=True)
    report = dict(manifest_sha256={k:sha(v) for k,v in paths.items()}, summary=summary,
        excluded_runs=failures, speedups=speedups,
        largest_tested_verified_within_20ms=largest,
        timing='16 sequential frames per configuration on one board; median and min–max, not a confidence interval.')
    (dest/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    with (dest/'summary.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    plt.style.use(STYLE)
    plt.rcParams.update({'pdf.fonttype':42, 'svg.fonttype':'none',
        'text.usetex':False, 'savefig.bbox':None})
    # Switzer after the shared style, which sets a serif stack; see common/switzer_font.py.
    from paper.common import switzer_font
    from paper.common import figstyle as sf
    plt.rcParams.update(switzer_font.rcparams())
    # House style copied from Figure 3 (build_dpose_combined.py): Title Case
    # labels at 9.5 pt, 9 pt ticks, no y spines but a light grid, grey dashed
    # reference rules named at the rule in 8.5 pt, bold 11 pt panel letters.
    REF, DASH, GRID, GRID_W, INK = 'black', (0, (4, 3)), '#DEDEDE', .55, '#232323'
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))
    fig.subplots_adjust(left=.1, right=.975, bottom=.19, top=.95, wspace=.33)
    # Learned encoder in grey: blue means continuous features in Figures 3 and 6.
    routes={'spar':('#d80032','o','SPAR'), 'learned':('#7a7a7a','s','Learned Tokenizer')}
    # Channel counts are swept densely to locate boundaries, which leaves the set
    # uneven. Plot a geometric ladder so the points are equally spaced on the log
    # axes, keeping each series' endpoints and the quoted limit. Every measured
    # configuration stays in panel B, the manifest and summary.csv.
    ladder={int(round(2**(k/2))) for k in range(21)} | {largest['optimized']['spar']}
    def shown(rows):
        ends={rows[0]['channels'],rows[-1]['channels']}
        keep=ladder|ends
        # 64 sits 0.25 % of a decade from the quoted 63 and would merge with it
        # on the rule; drop it from A wherever 63 is the labelled limit (B keeps it).
        if largest['optimized']['spar'] in {r['channels'] for r in rows}: keep-={64}
        return [r for r in rows if r['channels'] in keep]
    def series(name, route):
        return sorted((r for r in summary if r['implementation']==name and r['route']==route),
                      key=lambda r:r['channels'])

    # A: measured compute time per update against the deadline the hop sets.
    ax=axes[0]
    for name in paths:
        for route,(color,marker,label) in routes.items():
            rows=shown(series(name, route))
            n=np.array([r['channels'] for r in rows])
            y=np.array([r['median_ms'] for r in rows])
            low=y-np.array([r['min_ms'] for r in rows]); high=np.array([r['max_ms'] for r in rows])-y
            ax.plot(n,y,color=color,lw=1,ls='--' if name=='original' else '-',zorder=3)
            ax.errorbar(n,y,yerr=[low,high],fmt=marker,color=color,ms=3.6,elinewidth=.7,capsize=1.5,
                        mfc='white' if name=='original' else color,mec=color,mew=.65,zorder=4,
                        clip_on=False)
    decimal=FuncFormatter(lambda v,_: f'{v:,.0f}' if v>=1 else f'{v:g}')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(.85,1200); ax.set_ylim(.09,26000)
    ax.set_yticks([.1,1,10,100,1000,10000])
    ax.set_xlabel('EMG Channels',labelpad=5,fontsize=9.5)
    ax.set_ylabel('Compute Time per Update (ms)',labelpad=5,fontsize=9.5)
    ax.axhline(20,color=REF,ls=DASH,lw=.75,zorder=1)
    # '20 ms' as in Figures 3 and 6; the caption names it the update interval.
    ax.annotate('20 ms',(1,20),xytext=(2,3),textcoords='offset points',
                ha='left',va='bottom',fontsize=8.5,color=REF,zorder=5)
    # Name the two crossings the text quotes: the largest count whose every
    # measured update met the deadline. The next count misses it.
    # Both labels sit above the rule with a leader, clear of the legend: 23 up
    # and left of its point, 63 to the right of the optimized line.
    for name,offset,ha,rel in [('optimized',(30,7),'left',(0,.3)),('original',(-12,17),'right',(1,.3))]:
        n=largest[name]['spar']
        r=next(r for r in series(name,'spar') if r['channels']==n)
        ax.annotate(f'{n} Channels',(n,r['median_ms']),xytext=offset,textcoords='offset points',
                    ha=ha,va='center',fontsize=8.5,color=routes['spar'][0],zorder=5,
                    arrowprops=dict(arrowstyle='-',color=routes['spar'][0],lw=.6,shrinkA=1,shrinkB=3,relpos=rel))
    ax.grid(axis='both',color=GRID,lw=GRID_W); ax.set_axisbelow(True)
    handles=[Line2D([],[],color=c,marker=m,lw=0,ms=4,label=l) for c,m,l in routes.values()]
    handles+=[Line2D([],[],color=INK,marker='o',mfc='white',ms=3.6,ls='--',lw=1,label='Original Port'),
              Line2D([],[],color=INK,marker='o',ms=3.6,lw=1,label='Optimized Port')]
    ax.legend(handles=handles,loc='lower right',fontsize=8.5,frameon=True,facecolor='white',
              edgecolor='none',framealpha=1,labelspacing=.3,handletextpad=.5,
              borderaxespad=.2,borderpad=.3,handlelength=1.8)

    # B: the cost model, from every measured configuration. Per-channel cost is
    # flat once the montage fills, so the update time is proportional to channels.
    STAGES=[('frontend','Feature Extraction','#d80032','white'),
            ('quantizer','Codebook Search','#ef7f99',INK),
            ('preprocessing','Window + Normalize','#f9d2dc',INK)]
    opt=series('optimized','spar')
    N=np.array([r['channels'] for r in opt],float)
    total=np.array([r['median_ms'] for r in opt],float)
    # Least squares through the origin; an affine fit wants a negative intercept.
    fit=float((total*N).sum()/(N*N).sum())
    steady=N>=16
    worst=float((100*np.abs(fit*N[steady]-total[steady])/total[steady]).max())
    # Stage medians are taken separately; packing (<0.001 ms/channel) is not drawn.
    stage_sum=np.array([sum(r[k+'_median_ms'] for k in ['preprocessing','frontend','quantizer','packing'])
                        for r in opt])
    stage_gap=float((100*np.abs(stage_sum-total)/total).max())
    assert stage_gap<.5, stage_gap
    ax=axes[1]
    floor=np.zeros_like(N)
    at320=int(np.searchsorted(N,320))          # coefficients quoted where flat
    terms={}
    for key,label,shade,_ in STAGES:
        per=np.array([r[key+'_median_ms']/r['channels'] for r in opt])
        ax.fill_between(N,floor,floor+per,color=shade,lw=0,zorder=2)
        terms[key]=(floor[at320], per[at320])
        floor=floor+per
    ax.plot(N,total/N,color=INK,lw=.7,zorder=4)
    ax.plot(N,total/N,'o',color=INK,ms=1.8,mew=0,zorder=4)
    ax.axhline(fit,color=REF,ls=DASH,lw=.75,zorder=3)
    # T = aN fitted to the measured totals, drawn as its slope a.
    ax.annotate(f'Proportional Fit ({fit:.2f} ms)',(1,fit),xytext=(0,3),textcoords='offset points',
                ha='left',va='bottom',fontsize=8.5,color=REF,zorder=5)
    ax.annotate('Measured Total',(N[-1],total[-1]/N[-1]),xytext=(-8,3),textcoords='offset points',
                ha='right',va='bottom',fontsize=8.5,color=INK,zorder=5)
    # Thick bands hold their own labels; the thin one is named just under the stack top.
    for key,label,shade,ink in STAGES[:2]:
        base,height=terms[key]
        # Inset from the right edge so the units clear the panel border.
        ax.annotate(f'{label}  {height:.3f} ms',(1,base+.3*height),xycoords=('axes fraction','data'),
                    xytext=(-8,0),textcoords='offset points',ha='right',va='center',
                fontsize=8.5,color=ink,zorder=5)
    base,height=terms['preprocessing']
    ax.annotate(f'Window + Normalize  {height:.3f} ms',(70,base+height/2),xytext=(70,.385),
                ha='center',va='center',fontsize=8.5,color=INK,zorder=5,
                arrowprops=dict(arrowstyle='-',color=INK,lw=.6,shrinkA=1,shrinkB=0,relpos=(.5,0)))
    ax.set_xscale('log'); ax.set_xlim(.85,1200); ax.set_ylim(0,.41)
    ax.set_yticks([0,.1,.2,.3])
    ax.set_xlabel('EMG Channels',labelpad=5,fontsize=9.5)
    ax.set_ylabel('Compute Time per Channel (ms)',labelpad=5,fontsize=9.5)
    ax.grid(axis='y',color=GRID,lw=GRID_W); ax.set_axisbelow(True)

    for ax in axes:
        ax.xaxis.set_major_locator(FixedLocator([1,4,16,64,256,1024]))
        ax.xaxis.set_major_formatter(decimal)
        ax.spines[['left','right','top']].set_visible(False)
        sf.trim_spines(ax, which=('bottom',), offset=0)
        ax.tick_params(axis='x',labelsize=9,length=3,top=False,direction='out')
        ax.tick_params(axis='y',labelsize=9,length=0,pad=4,right=False)
        ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
    axes[0].yaxis.set_major_formatter(decimal)
    # AFTER trim_spines, which regenerates the tick labels: keep the wide end
    # labels inside their own panel.
    for ax in axes:
        labels=ax.get_xticklabels(); labels[-1].set_horizontalalignment('right')
    # Letters as in the Affinity-finished figures: Switzer Regular 12 pt at the top edge.
    fig.text(.004,1.0,'A',fontsize=12,va='top')
    fig.text(axes[1].get_position().x0-.07,1.0,'B',fontsize=12,va='top')
    assert not any(a.get_title() for a in axes) and not sf.check_font_sizes(fig, hi=12), sf.check_font_sizes(fig, hi=12)

    for ext in ['pdf','svg','png']:
        fig.savefig(dest/f'stm32n6-optimized.{ext}',dpi=300,bbox_inches=None)
    plt.close(fig)

    # Every number in the caption is computed here and printed below, so the
    # caption can be checked against summary.json line by line.
    edge=next(r for r in opt if r['channels']==largest['optimized']['spar'])
    over=next(r for r in opt if r['channels']>edge['channels'])
    margin=100*(20-edge['max_ms'])/20
    learned_one=next(r for r in series('optimized','learned') if r['channels']==1)
    assert learned_one['min_ms']>20 and all(r['min_ms']>20 for r in summary if r['route']=='learned')
    numbers=dict(largest_exact=largest_exact, largest_exact_timing=largest_exact_timing,
                 fit_ms_per_channel=fit, worst_pct_n16=worst, stage_gap_pct=stage_gap,
                 per_channel_at_320={k:v[1] for k,v in terms.items()},
                 largest=largest, edge_max_ms=edge['max_ms'], margin_pct=margin,
                 next_channels=over['channels'], next_max_ms=over['max_ms'],
                 learned_one_channel_ms=learned_one['median_ms'],
                 configurations=len(summary), disagreeing=len(failures))
    # The labelled counts must be the timing boundary too, not only the agreement one.
    for name in paths:
        rows=series(name,'spar'); n=largest[name]['spar']
        assert all(r['all_frames_within_20ms'] for r in rows if r['channels']<=n)
        assert not next(r for r in rows if r['channels']>n)['all_frames_within_20ms']
    caption=(r'\textbf{SPAR tokenization on a Cortex-M55 microcontroller meets the 20\,ms update interval up to '
        +f'{largest["optimized"]["spar"]}'+r' channels.} '
        r'Both C ports ran the frozen tokenizers on one NUCLEO-N657X0-Q (600\,MHz, FP32, caches enabled, '
        r'no neural accelerator) with synthetic 2\,kHz input and 160\,ms windows. One update comprises the '
        r'window shift, normalization, feature extraction, exhaustive 1,024-code assignment and 10-bit packing; '
        r'acquisition, BLE transmission and pose decoding are excluded. '
        r'\textbf{A:} compute time per update versus channel count for the optimized (filled markers, solid '
        r'lines) and original ports (open markers, dashed lines); a geometric subset of the measured counts '
        r'is shown. The dashed horizontal line marks the 20\,ms update interval. Labels give the largest '
        r'channel count for which every update finished within this interval ('
        +f'{largest["optimized"]["spar"]} optimized, {largest["original"]["spar"]} original'
        r'). The learned tokenizer required '+f'{learned_one["median_ms"]:.0f}'+r'\,ms for a single channel. '
        r'Markers show medians over 16 updates after 16 warm-up updates; minimum--maximum whiskers are '
        r'mostly hidden by the markers. '
        r'\textbf{B:} per-channel compute time of the optimized SPAR port by processing stage; stage labels '
        r'give the per-channel cost at 320 channels. The dashed line is a least-squares fit through the '
        r'origin, within '+f'{np.ceil(worst):.0f}'+r'\,\% of every median from 16 channels. '
        r'Capacities are observed over 16 updates rather than guaranteed bounds and also require token '
        r'agreement with the host in every update; at '+f'{edge["channels"]}'+r' channels the slowest update '
        r'was '+f'{margin:.2f}'+r'\,\% below the interval. The optimized port is not bit-identical to the '
        r'released tokenizer; a bit-exact build reaches '+str(largest_exact)+r' channels (Appendix~\ref{app:n6}).')
    plain=caption.replace(r'\,',' ').replace(r'\%','%').replace(r'\textbf{','').replace('}','') \
                 .replace(r'Appendix~\ref{app:n6','appendix')
    (dest/'caption.txt').write_text(plain+'\n')
    # A macro, not bare text: \input inside \caption is a fragile argument.
    (dest/'caption.tex').write_text('\\newcommand{\\stmnsixcaption}{'+caption+'}\n')
    (dest/'include.tex').write_text('\\begin{figure*}[t]\n\\centering\n'
        '\\includegraphics[width=\\textwidth]{figures/stm32n6-optimized.pdf}\n'
        '\\caption{\\stmnsixcaption}\n\\label{fig:stm32n6}\n\\end{figure*}\n')
    print(json.dumps(numbers,indent=2))
    print(dest/'stm32n6-optimized.pdf')


if __name__=='__main__': main()
