"""Add audited full-cohort SPAR results to the existing HD-EMG figure.

The 54 existing participant contrast vectors are retained; twelve declared
contrasts extend the joint max-T family to 66. No training.
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import shutil

import numpy as np

CAMPAIGN = 'hdemg_spar_simple_full_v1'
TOKEN = (CAMPAIGN, 'tokens')
FEATURE = (CAMPAIGN, 'continuous')
OLD_TOKEN = ('hdemg_v20_v1', 'tokens')
OLD_FEATURE = ('hdemg_v20_continuous_v1', 'continuous')
PEOPLE = list(range(1, 10))
SEEDS = [260911, 260912, 260913]
TASKS = [2, 4, 6, 8, 10, 15, 18, 19]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def extend(b, rows, formats, benchmark, analysis):
    root = b.ROOT
    run = root/'results'/CAMPAIGN
    results = json.loads((run/'results.json').read_text())
    cost_path = run/'resource-benchmark.json'
    costs = json.loads(cost_path.read_text())
    assert results['complete'] and results['fits'] == 54 and costs['complete']
    assert costs['torch_version'].split('+')[0] == benchmark['dependencies']['torch'].split('+')[0]
    assert costs['token_index_cast_included']
    inputs_path = run/'subject_01/figure-video-inputs.npz'
    assert b.resources.digest(inputs_path) == costs['source_sha256']
    with np.load(inputs_path) as inputs:
        for key, expected in benchmark['inputs']['reference']['identities'].items():
            np.testing.assert_array_equal(inputs['val_'+key], expected)
    assert results['subjects'] == PEOPLE and results['seeds'] == SEEDS
    assert results['protocol_sha256'] == b.resources.digest(run/'protocol.json')
    sources = {str(p.relative_to(root)):b.resources.digest(p) for p in
               [run/'results.json', run/'protocol.json', run/'local-audit.json', cost_path]}
    by_key = {(r['campaign'], r['arm']):r for r in rows}
    task_means = {}
    for arm, old in [('tokens', OLD_TOKEN), ('continuous', OLD_FEATURE)]:
        resource, = [r for r in costs['models'] if r['arm'] == arm]
        scores = [r for r in results['scores'] if r['arm'] == arm]
        assert len(scores) == 27 and resource['parameter_count'] == 689468
        assert resource['checkpoint_sha256'] == b.resources.digest(run/f'subject_01/fits/{arm}_260911/best.pt')
        assert len(resource['samples_ms']) == resource['measured_calls'] == 48
        seed_values = {str(s):{} for s in PEOPLE}
        task_values = np.empty((9, 3, 8))
        for score in scores:
            s, seed = score['subject'], score['seed']
            path = run/f'subject_{s:02d}/fits/{arm}_{seed}/test_predictions.npz'
            assert b.resources.digest(path) == score['predictions_sha256']
            source = np.load(path)
            values = np.linalg.norm((source['prediction']-source['target']).reshape(16,50,20,3),axis=-1)
            np.testing.assert_allclose(values.mean(), score['error_mm'], rtol=0, atol=2e-6)
            np.testing.assert_allclose(score['existing_learned_error_mm'],
                by_key[old]['error_participant_seed_values'][s][seed], rtol=0, atol=2e-5)
            assert set(source['tasks']) == set(TASKS)
            for t in TASKS:
                assert sum(source['tasks']==t) == 2
                np.testing.assert_allclose(values[source['tasks']==t].mean(), score['per_task'][str(t)], rtol=0, atol=2e-6)
            seed_values[str(s)][str(seed)] = score['error_mm']
            task_values[s-1, SEEDS.index(seed)] = [score['per_task'][str(t)] for t in TASKS]
            sources[str(path.relative_to(root))] = score['predictions_sha256']
        task_means[arm] = task_values.mean(axis=1)
        means = {s:float(np.mean(list(v.values()))) for s,v in seed_values.items()}
        row = dict(campaign=CAMPAIGN, arm=arm, representation=arm, family='SPAR', panel='B',
            label='Proposed spatial CNN + TCN + SPAR '+('tokens' if arm=='tokens' else 'features'),
            plot_label='SPAR '+('tokens' if arm=='tokens' else 'features'),
            experiment_group_id='proposed', experiment_group='Proposed spatial CNN + TCN',
            error_mean=float(np.mean(list(means.values()))), error_participant_means=means,
            error_participant_seed_values=seed_values, error_n_participants=9, error_n_seeds=3,
            error_evaluation='chronological_test', error_units='mm', error_precision='FP32; TF32 disabled',
            error_scope='Same-session/task chronological test; same 20 wrist-relative landmarks and all 9 participants.',
            model_parameters_million=resource['parameter_count']/1e6, timing_n_calls=48,
            resource_scope=costs['scope'], resource_session={k:costs[k] for k in ['gpu','cpu_threads','torch_version']},
            subject=1, seed=260911, resource_campaign=CAMPAIGN,
            model_literature={'relationship':'Custom 320-channel finite-grid adaptation of SPAR_SIMPLE, with a train-only per-participant Lloyd quantizer and our spatial CNN + causal TCN.'},
            **{k:v for k,v in resource.items() if k!='arm'})
        row['arm'] = arm
        rows.append(row)
    by_key = {(r['campaign'], r['arm']):r for r in rows}
    b.GROUPS[-1]['variants'] += [TOKEN, FEATURE]
    for key in [OLD_TOKEN, OLD_FEATURE, TOKEN, FEATURE]:
        by_key[key]['plot_label'] = b.plot_model_label(by_key[key])

    formats[0]['label'] = 'Tokens (Both)'
    formats[1]['label'] = 'Raw EMG'
    formats[2]['label'] = 'Learned Features'
    formats.append(dict(representation='continuous', encoding='FP32', label='SPAR Features',
        items_per_second=320*18*50, bits_per_item=32, kbit_s=320*18*50*32/1000,
        mb_min=320*18*50*4*60/1e6, structural_zero_slots_per_frame=504))
    benchmark['spar_extension'] = costs
    absolute = np.abs(task_means['tokens']-task_means['continuous']).mean(axis=1)
    summary = dict(decoder='SPAR', table_label='Proposed (SPAR)', mean_absolute_change_mm=float(absolute.mean()),
        sd_across_participants_mm=float(absolute.std(ddof=1)), n_participants=9,
        participant_means_mm={str(s):float(v) for s,v in zip(PEOPLE,absolute)})
    b.ABSOLUTE_ANALYSIS['summaries'].append(summary)
    b.REPRESENTATION_PAIRS.append(('SPAR', TOKEN, FEATURE))
    b.REPRESENTATION_TABLE_LABELS.update(Proposed='Proposed (Learned)', SPAR='Proposed (SPAR)')

    # Save the scope before calculating any new p-values.
    definitions = []
    for ref in [b.L1_CNN_KEY, b.MATCHED_CNN_KEY, b.tbme.KEY]:
        for key in [TOKEN, FEATURE]:
            definitions.append(dict(kind='cnn_reference', alternative=list(key), reference=list(ref)))
    definitions += [dict(kind='spar_representation', alternative=list(TOKEN), reference=list(FEATURE)),
                    dict(kind='spar_vs_learned_tokens', alternative=list(TOKEN), reference=list(OLD_TOKEN))]
    definitions += [dict(kind='absolute_error_change_spar', alternative=['absolute_error_change', d],
                         reference=['absolute_error_change','SPAR']) for d in ['TCN','Transformer','Qwen3','Proposed']]
    protocol = dict(family_size=66, retained_analysis='results/hdemg_absolute_change_analysis.json',
        retained_analysis_sha256=b.resources.digest(b.ABSOLUTE_RECORD), retained_contrasts=54,
        additions=definitions, independent_unit='participant', participants=PEOPLE, seeds=SEEDS, tasks=TASKS,
        test='Two-sided exact synchronized participant sign-flip studentized max-T; 512 sign patterns.',
        absolute_endpoint='First average each arm over 2 windows and 3 seeds within participant/task; take the absolute token-feature difference; then average 8 tasks within participant.',
        scope='Extends the family with the declared SPAR contrasts.')
    protocol_path = root/'results/hdemg_spar_figure_protocol.json'
    if protocol_path.exists(): assert json.loads(protocol_path.read_text()) == protocol
    else: write(protocol_path, protocol)
    comparisons = copy.deepcopy(analysis['groups'][0]['comparisons'])
    summaries = {s['decoder']:s for s in b.ABSOLUTE_ANALYSIS['summaries']}
    def values(key):
        source = summaries[key[1]]['participant_means_mm'] if key[0]=='absolute_error_change' else by_key[tuple(key)]['error_participant_means']
        return np.array([source.get(str(s),source.get(s)) for s in PEOPLE])
    def label(key):
        return b.REPRESENTATION_TABLE_LABELS[key[1]]+' absolute error change' if key[0]=='absolute_error_change' else by_key[tuple(key)]['label']
    for index, definition in enumerate(definitions):
        alt, ref = definition['alternative'], definition['reference']
        d = values(alt)-values(ref)
        pair = dict(contrast_kind=definition['kind'], alternative=alt, reference=ref, a=alt, b=ref,
            i=100+index, j=0, label_a=label(alt), label_b=label(ref),
            effect_alternative_minus_reference_mm=float(d.mean()), mean_difference_a_minus_b=float(d.mean()),
            participant_differences={str(s):float(v) for s,v in zip(PEOPLE,d)},
            effect_ci95_pointwise_mm=b.stats.exact_location_interval(d), n_participants=9,
            n_seeds_per_participant=3, permutations=512,
            leave_one_participant_out_effects_mm={str(s):float(np.delete(d,s-1).mean()) for s in PEOPLE})
        if definition['kind'] != 'cnn_reference':
            pair.update(contrast_id=f'spar_extension_{index:02d}', display_label=alt[1])
        comparisons.append(pair)
    matrix = np.array([[p['participant_differences'][str(s)] for p in comparisons] for s in PEOPLE])
    assert matrix.shape == (9,66)
    observed, raw, adjusted, null = b.stats.exact_max_t(matrix)
    for i,pair in enumerate(comparisons):
        if i<54:
            assert raw[i] == pair['p_raw'] and adjusted[i] >= pair['p_adjusted']
            pair['p_adjusted_previous_54'] = pair['p_adjusted']
        pair.update(t_observed=float(observed[i]),p_raw=float(raw[i]),p_adjusted=float(adjusted[i]),
            p_maxT=float(adjusted[i]),significant=bool(adjusted[i]<.05))
    analysis['groups'][0].update(comparisons=comparisons, family_size=66)
    analysis.update(family_size=66, comparison_protocol=protocol, maxT_reference_distribution=null.tolist(),
        significant_comparisons=sum(p['significant'] for p in comparisons),
        minimum_observed_adjusted_p=float(min(adjusted)), sources=sources,
        method=protocol['test']+' Requires independent participants and joint sign symmetry of true-null contrast vectors. '+protocol['scope'],
        spar_absolute_summary=summary, spar_task_means={k:v.tolist() for k,v in task_means.items()})
    assert len(rows)==19 and sum(r['error_n_participants']*r['error_n_seeds'] for r in rows)==513
    return rows, formats, benchmark, analysis


def caption(b, supplement=False):
    if supplement:
        return ('Smaller HD-EMG decoder baselines. Errors use the same personalized, same-session/task chronological test split, 20 wrist-relative 3D landmarks and nine participants as the main figure. Tokenizer pretraining excluded the validation and test windows. '
            'Gray points average three seeds within participant; colored symbols average participants. The dashed line shows the TBME CNN retrained on this split. '
            'Parameter counts include receiver parameters and exclude upstream feature extraction/tokenization. Latency shows the mean and sample SD of 48 FP32 decoder calls on an RTX 4090, batch one, after 12 warmups over four validation inputs, with full context recomputation. '
            'Branches compare each baseline with the adapted L1 CNN in the main figure; that reference is outside this view. '
            'All tests use the same 66-contrast, two-sided exact max-T family as the main figure, with 512 synchronized participant sign patterns. '
            'The procedure requires independent participants and joint null sign symmetry. '
            '* p<0.05, ** p<0.01, *** p<0.001; n.s. denotes non-rejection, not equivalence. Full statistical tables accompany the main figure.')
    start = ('Smaller HD-EMG decoder baselines, retained in the supplement. ' if supplement else
        'HD-EMG decoding with learned EMG representations and SPAR features. (a) Pose error, receiver parameter count and decoder latency. ')
    text = start + ('All errors use the same personalized, same-session/task chronological test split, 20 wrist-relative 3D landmarks and nine participants; gray points average three seeds within participant, and colored symbols average participants. Tokenizer pretraining excluded the validation and test windows. '
        'The dashed line is the TBME CNN retrained on this split. The adapted CNN uses L1 loss and weight averaging; matched TCN/Transformer models share their declared training setup. '
        'The Qwen3-0.6B backbone is frozen, with supervised EMG adapters; bypass and random controls are included. '
        'SPAR uses six bands and finite-grid local log-SPD features. A new 1,024-code Lloyd quantizer is fitted on each participant’s training features and frozen across three decoder seeds. '
        'Only SPAR tokens versus SPAR features isolates their quantization; learned and SPAR front ends also differ in features and pretraining. '
        'Parameters include frozen backbone parameters but exclude buffers and upstream feature extraction/tokenization. Latency is decoder-only FP32 on an RTX 4090, batch one, full context recomputation: mean and sample SD of 48 calls after 12 warmups over four validation inputs. Costs use participant 1, seed 260911; Qwen costs use the earlier pilot checkpoint of the same verified architecture. Accuracy uses the fitted model precision (FP16/BF16/FP32); all costs standardize FP32. ')
    if not supplement:
        text += ('(b) Array payload at 50 feature frames/s: uint16 tokens for both front ends, 2,048-Hz int16 raw EMG, and FP32 learned/SPAR features. SPAR includes all 5,760 dense values, including 504 structural zeros; 5,256 unique values would permit a smaller payload. This is stream arithmetic, excluding container overhead. '
            '(c) Participant means of absolute within-task token–feature error changes: average two windows and three seeds within task, take the absolute difference, then average eight tasks. Branches compare each row with SPAR; lower means a smaller change, not necessarily lower pose error. ')
    return text + ('Branches in the error panel compare with the adapted CNN. All displayed tests use one family of 66 two-sided exact max-T comparisons, retaining the previous 54 contrasts and adding 12 SPAR contrasts. '
        'The 512 synchronized participant sign patterns require independent participants and joint null sign symmetry. '
        '* p<0.05, ** p<0.01, *** p<0.001; n.s. denotes non-rejection, not equivalence. Complete effects, pointwise intervals and p-values accompany the figure.')


def export(b, rows, formats, benchmark, analysis, supplement=False):
    selected = b.selected_rows(rows, b.EARLY_GROUPS if supplement else b.GROUPS)
    stem = b.SUPPLEMENT_STEM if supplement else b.STEM
    fig = b.make_figure(selected, formats, analysis, supplement=supplement)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in [*fig.texts, *[t for ax in fig.axes for t in ax.get_xticklabels()+ax.get_yticklabels()+[ax.xaxis.label]]]:
        box = item.get_window_extent(renderer)
        if item.get_visible() and item.get_text():
            assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.width+1 and box.y1<=fig.bbox.height+1, item.get_text()
    paths = [b.PDFS/f'{stem}.pdf', b.OUT/f'{stem}.svg', b.OUT/f'{stem}.png']
    fig.savefig(paths[0]); fig.savefig(paths[1]); fig.savefig(paths[2], dpi=400, facecolor='white')
    b.plt.close(fig)
    prose = caption(b,supplement)
    (b.OUT/f'{stem}.md').write_text(prose+'\n\nReproduce: `uv run python -m paper.figures.build_hdemg_combined`.\n')
    (b.OUT/f'{stem}.tex').write_text('\\begin{figure*}[t]\n\\centering\n'
        f'\\includegraphics[width=\\textwidth]{{output/pdf/{stem}.pdf}}\n'
        '\\caption{'+b.latex_caption(prose).replace('p<0.05',r'$p<0.05$').replace('p<0.01',r'$p<0.01$').replace('p<0.001',r'$p<0.001$')+'}\n'+f'\\label{{fig:{stem}}}\n\\end{{figure*}}\n')
    record = dict(complete=True, variant_count=len(selected), decoder_rows=selected,
        all_source_test_participant_seed_cells_verified=513,
        test_participant_seed_cells_verified=27*len(selected), timed_calls_verified=48*len(selected),
        figure_width_in=b.FIGURE_WIDTH_IN, figure_height_in=b.SUPPLEMENT_HEIGHT_IN if supplement else b.FIGURE_HEIGHT_IN,
        display_groups=b.EARLY_GROUPS if supplement else b.GROUPS, baseline_reference=b.BASELINE_REFERENCE,
        storage_formats=None if supplement else formats, statistics=analysis,
        significance_annotations=b.significance_brackets(selected,b.EARLY_GROUPS if supplement else b.GROUPS,analysis),
        paired_representation_differences=None if supplement else b.paired_representation_data(selected),
        paired_significance_annotations=None if supplement else b.paired_significance(selected,analysis),
        caption=prose, sources={**b.accuracy.SOURCES,**analysis['sources']},
        builder_sha256=b.resources.digest(Path(b.__file__)), extension_sha256=b.resources.digest(Path(__file__)),
        benchmark=benchmark, outputs=[str(p.relative_to(b.ROOT)) for p in paths])
    write(b.OUT/f'{stem}.json',record)
    flat = [dict(campaign=r['campaign'],arm=r['arm'],label=r['label'], error_mean_mm=r['error_mean'],
        participant_sd_mm=float(np.std(list(r['error_participant_means'].values()),ddof=1)),
        parameters=r['parameter_count'],latency_mean_ms=r['latency_mean_ms'],latency_sd_ms=r['latency_sd_ms']) for r in selected]
    with (b.OUT/f'{stem}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    table=[r'\begin{tabular}{lrrr}\toprule',r'Decoder & Error (mm) & Parameters (M) & Latency (ms)\\',
        r' & Mean $\pm$ SD & & Mean $\pm$ SD\\\midrule']
    for group in (b.EARLY_GROUPS if supplement else b.GROUPS):
        table.append(r'\multicolumn{4}{l}{\textbf{'+group['label']+r'}}\\')
        for row in b.selected_rows(selected,[group]):
            label=b.plot_model_label(row)
            if row['experiment_group_id']!='proposed': label+=' ('+row['representation']+')'
            sd=float(np.std(list(row['error_participant_means'].values()),ddof=1))
            table.append(f"{label} & {row['error_mean']:.2f} $\\pm$ {sd:.2f} & {row['model_parameters_million']:.3f} & {row['latency_mean_ms']:.3f} $\\pm$ {row['latency_sd_ms']:.3f}"+r'\\')
    table += [r'\bottomrule\end{tabular}']
    (b.OUT/f'TABLE_{stem}.tex').write_text('\n'.join(table)+'\n')


def run(b):
    rows, formats, benchmark = b.load_data()
    analysis = b.reference_statistics(rows)
    rows, formats, benchmark, analysis = extend(b, rows, formats, benchmark, analysis)
    for supplement in [False,True]: export(b,rows,formats,benchmark,analysis,supplement)
    write(b.ROOT/'results/hdemg_figure_analysis.json',analysis)
    write(b.ROOT/'results/hdemg_spar_figure_analysis.json',analysis)
    write(b.OUT/'hdemg-figure-statistics.json',analysis)
    tests = analysis['groups'][0]['comparisons']
    records = [dict(alternative=p['label_a'], reference=p['label_b'],kind=p['contrast_kind'],
        effect_mm=p['effect_alternative_minus_reference_mm'], ci95_low_mm=p['effect_ci95_pointwise_mm'][0],
        ci95_high_mm=p['effect_ci95_pointwise_mm'][1],p_raw=p['p_raw'],p_adjusted_66=p['p_adjusted'],
        symbol=b.significance_symbol(p['p_adjusted']) or 'n.s.') for p in tests]
    with (b.OUT/'hdemg-figure-statistics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    lines=['# HD-EMG figure: expanded SPAR analysis','',analysis['method'],'',
           'All 54 previous participant vectors, effects and raw p-values are retained. The twelve additions were declared in `results/hdemg_spar_figure_protocol.json` before calculating their joint adjusted p-values. The earlier two-contrast SPAR Holm report remains a separate analysis.','',
           '| Alternative | Reference | Effect (mm) | Pointwise 95% interval | Raw p | Joint 66 max-T p |','|---|---|---:|---|---:|---:|']
    lines += [f"| {r['alternative']} | {r['reference']} | {r['effect_mm']:.3f} | [{r['ci95_low_mm']:.3f}, {r['ci95_high_mm']:.3f}] | {r['p_raw']:.6f} | {r['p_adjusted_66']:.6f} |" for r in records]
    report='\n'.join(lines)+'\n'
    (b.ROOT/'output/HDEMG_SPAR_FIGURE_ANALYSIS.md').write_text(report)
    (b.OUT/'hdemg-figure-statistics.md').write_text(report)
    for suffix in ['json','csv','md']:
        shutil.copy2(b.OUT/f'hdemg-figure-statistics.{suffix}',b.OUT/f'hdemg-statistical-tests.{suffix}')
    table=[r'% Requires booktabs and longtable. All 66 joint max-T comparisons.',
        r'\begin{longtable}{p{2.1in}p{2.1in}rrr}\toprule',
        r'Alternative & Reference & $\Delta$ (mm) & Pointwise 95\% CI & Max-T $p$\\\midrule\endhead']
    for r in records:
        table.append(f"{r['alternative']} & {r['reference']} & {r['effect_mm']:+.3f} & $[{r['ci95_low_mm']:+.3f}, {r['ci95_high_mm']:+.3f}]$ & {r['p_adjusted_66']:.6f}"+r'\\')
    table.append(r'\bottomrule\end{longtable}')
    (b.OUT/'TABLE_hdemg-statistical-tests.tex').write_text('\n'.join(table)+'\n')
    (b.OUT/'hdemg-statistical-tests.tex').write_text('% Requires booktabs and longtable; nine independent participants, 66 comparisons.\n'
        r'\input{output/figures/TABLE_hdemg-statistical-tests.tex}'+'\n')
    for name in ['build_hdemg_combined.py','hdemg_spar_figure_extension.py']:
        shutil.copy2(b.ROOT/'scripts'/name,b.OUT/name)
    print(json.dumps(dict(main_rows=13,supplement_rows=4,statistical_family=66,
        spar=[r for r in records if r['kind'].startswith('spar')]),indent=2))
