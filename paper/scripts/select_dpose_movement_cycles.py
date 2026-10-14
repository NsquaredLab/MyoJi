"""Select two complete kinematic cycles before reading any model predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.signal import find_peaks, savgol_filter

import paths

TASKS={2:'Thumb',4:'Index',6:'Middle',8:'Ring',10:'Little',15:'Fist',18:'Two-finger pinch',19:'Three-finger pinch'}
FINGERS={2:[0],4:[1],6:[2],8:[3],10:[4],15:[0,1,2,3,4],18:[0,1],19:[0,1,2]}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=paths.RESULTS/'dpose_movement_video');args=p.parse_args();root=args.root
    # The source protocol is the frozen within-person protocol; its 'files' pins the kinematics archive.
    protocol=json.loads((paths.REFERENCE/'dpose_within_v1/protocol.json').read_text())
    path=paths.HDEMG/'Sub1_kinematics.pkl'
    assert digest(path)==protocol['files'][path.name]
    kin=joblib.load(path);segments=[];arrays={}
    for task,name in TASKS.items():
        y=kin[str(task)];count=y.shape[-1]*125//128;train_end=count*3//5
        ends=np.arange(39,count,40);source_indices=(ends*128)//125
        points=y[...,source_indices].transpose(2,0,1).astype(np.float32)
        truth=points[:,1:]-points[:,:1]
        ids=[j for f in FINGERS[task] for j in range(f*4,f*4+4)]
        train=ends<train_end;v=truth[train][:,ids,:].reshape(train.sum(),-1)
        v=v-v.mean(axis=0);_,vectors=np.linalg.eigh(v.T@v);axis=vectors[:,-1]
        if axis[np.argmax(abs(axis))]<0:axis=-axis
        signal=v@axis;smoothed=savgol_filter(signal,21,3)
        span=float(np.percentile(smoothed,95)-np.percentile(smoothed,5))
        peaks,properties=find_peaks(smoothed,prominence=.25*span,distance=40)
        eligible=peaks[(peaks>=200)&(peaks<len(v)-100)]
        triples=[eligible[i:i+3] for i in range(len(eligible)-2)
                 if np.diff(eligible[i:i+3]).min()>=50 and np.diff(eligible[i:i+3]).max()/np.diff(eligible[i:i+3]).min()<=1.8]
        if not triples:raise ValueError((task,'No two complete cycles under fixed criteria',peaks.tolist()))
        a,m,b=map(int,triples[0]);first=((a-50)//50)*50;last=(b//50)*50
        starts=[(frame-100)*40 for frame in range(first,last+1,50)]
        assert min(starts)>=0 and max(starts)+6000<=train_end
        segment=dict(task=task,name=name,source_split='train',train_end_2000hz_sample=int(train_end),
            cycle_peak_frames=[a,m,b],cycle_peak_times_seconds=(ends[[a,m,b]]/2000).tolist(),
            cycle_durations_seconds=(np.diff([a,m,b])/50).tolist(),
            display_start_frame=a,display_end_frame_inclusive=b,display_frames=b-a+1,
            inference_first_global_frame=first,inference_last_global_frame=last+49,
            inference_window_starts_2000hz=starts,warmup_frames=a-first,
            selection_signal='First principal component of the instructed fingers’ wrist-relative landmarks, computed on the chronological training block.',
            candidate_peak_frames=peaks.tolist())
        segments.append(segment)
        arrays[f'task_{task}_truth']=truth.astype(np.float32)
        arrays[f'task_{task}_ends']=ends
        arrays[f'task_{task}_selection_signal']=signal
        arrays[f'task_{task}_selection_smoothed']=smoothed
        print(json.dumps({k:segment[k] for k in ['task','name','cycle_peak_times_seconds','cycle_durations_seconds','display_frames','warmup_frames']}),flush=True)
    receipt=dict(subject=1,seed=260911,on_screen_title='Movement demonstration',source_split='train',
        policy='First eligible pair of full cycles selected from ground-truth kinematics only; no prediction/error-based segment selection.',
        peak_rules=dict(minimum_context_seconds=4,savgol_frames=21,savgol_order=3,prominence_fraction_of_5_to_95_percentile_span=.25,
                        minimum_cycle_seconds=1,maximum_cycle_duration_ratio=1.8),
        source_file_sha256=digest(path),selection_code_sha256=digest(Path(__file__)),segments=segments)
    root.mkdir(parents=True,exist_ok=True)
    (root/'segment-selection.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    np.savez_compressed(root/'kinematic-cycle-selection.npz',**arrays)


if __name__=='__main__':main()
