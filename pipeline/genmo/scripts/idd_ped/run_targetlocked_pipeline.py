#!/usr/bin/env python3
"""Re-run IDD-PeD joint reconstruction with annotation-locked identities."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.idd_ped.camera_geometry import save_camera_npz  # noqa:E402
from scripts.idd_ped.target_annotation import (load_target_track,save_target_npz,
                                                write_target_video_and_overlay)  # noqa:E402


def run(cmd:list[str]):
    print('+',' '.join(cmd),flush=True); subprocess.run(cmd,cwd=ROOT,check=True)


def clip_id(entry): return entry["output"].split('_',1)[0]


def repair_and_rebase_camera(source:Path, full_out:Path, target_out:Path, track:dict, cid:str):
    with np.load(source,allow_pickle=False) as z: old={k:z[k] for k in z.files}
    T=old["T_c2w"].astype(np.float64).copy(); frames=old["frame_numbers"].astype(np.int64)
    corrected=[]
    requested=[8502,8514] if cid=='04' else []
    for sf in requested:
        found=np.flatnonzero(frames==sf)
        if len(found)!=1 or found[0]==0 or found[0]==len(T)-1: raise ValueError(f'Cannot correct source frame {sf}')
        i=int(found[0]); alpha=(sf-frames[i-1])/float(frames[i+1]-frames[i-1])
        T[i,:3,3]=(1-alpha)*T[i-1,:3,3]+alpha*T[i+1,:3,3]
        slerp=Slerp([0,1],Rotation.from_matrix(np.stack([T[i-1,:3,:3],T[i+1,:3,:3]])))
        T[i,:3,:3]=slerp([alpha]).as_matrix()[0]; corrected.append(sf)
    start_local=int(track["local_frames"][0]); anchor=np.linalg.inv(T[start_local])
    T=np.einsum('ij,njk->nik',anchor,T).astype(np.float32); Tw=np.linalg.inv(T).astype(np.float32)
    confidence=old["tracking_confidence"].copy(); failed=old["tracking_failed"].copy()
    for sf in corrected:
        i=int(np.flatnonzero(frames==sf)[0]); failed[i]=False
        confidence[i]=min(confidence[i-1],confidence[i+1])
    meta=json.loads(str(old["metadata_json"])); meta.update({
        "normalization_origin":"target_first_frame","normalization_source_frame":int(track["source_frames"][0]),
        "corrected_source_frames":corrected,"correction_method":"SE3 neighbor interpolation",
    })
    intr=old["intrinsics"]
    save_camera_npz(full_out,T,Tw,intr,frames,old["timestamps"],confidence,failed,float(old["fps"]),meta)
    idx=track["local_frames"].astype(int)
    tmeta={**meta,"scope":"target_track","bbox_fingerprint":track["fingerprint"],
           "target_key":[track["set_id"],track["video_id"],track["pedestrian_id"]]}
    save_camera_npz(target_out,T[idx],Tw[idx],intr,track["source_frames"],
                    track["source_frames"]/float(old["fps"]),confidence[idx],failed[idx],float(old["fps"]),tmeta)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',type=Path,default=ROOT/'inputs/idd-ped_inputs')
    p.add_argument('--database',type=Path,default=None)
    p.add_argument('--source_outputs',type=Path,default=ROOT/'outputs/idd_ped_joint')
    p.add_argument('--output_root',type=Path,default=ROOT/'outputs/idd_ped_joint_targetlocked')
    p.add_argument('--clips',nargs='+',default=['01','03','02','04','05'])
    p.add_argument('--ckpt',type=Path,default=ROOT/'inputs/pretrained/gem_smpl.ckpt')
    p.add_argument('--hmr2_ckpt',type=Path,default=ROOT/'inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt')
    args=p.parse_args(); db=args.database or args.inputs/'iddp_database.pkl'
    if not db.is_file(): raise FileNotFoundError(f'Required annotation database missing: {db}')
    manifest_path=args.inputs/'selection_manifest.json'; manifest=json.loads(manifest_path.read_text())
    args.output_root.mkdir(parents=True,exist_ok=True); summary=[]
    for cid in args.clips:
        entry=next(x for x in manifest['clips'] if clip_id(x)==cid)
        video=args.inputs/entry['output']; track=load_target_track(db,entry)
        out=args.output_root/video.stem; out.mkdir(parents=True,exist_ok=True)
        try:
            save_target_npz(out/'target_track.npz',track)
            target_video=out/'target_track_fullframe.mp4'
            write_target_video_and_overlay(video,track,target_video,out/'target_bbox_overlay.mp4',float(manifest['fps']))
            old_dirs=sorted(x for x in args.source_outputs.glob(f'{cid}_*') if x.is_dir())
            if len(old_dirs)!=1: raise FileNotFoundError(f'Expected one existing DROID output for {cid}')
            repair_and_rebase_camera(old_dirs[0]/'camera_trajectory.npz',out/'camera_trajectory.npz',
                                     out/'camera_trajectory_target.npz',track,cid)
            cache_name=f"{track['pedestrian_id']}_{track['fingerprint'][:16]}"
            demo_root=out/f'_genmo_{cache_name}'
            run([sys.executable,'scripts/demo/demo_smpl_hpe.py','--video',str(target_video),
                 '--camera_traj',str(out/'camera_trajectory_target.npz'),'--target_track',str(out/'target_track.npz'),
                 '--ckpt_path',str(args.ckpt),'--hmr2_ckpt',str(args.hmr2_ckpt),
                 '--output_root',str(demo_root),'--reprojection_only'])
            generated=demo_root/target_video.stem
            cache_dir=out/'preprocess_target'/cache_name; cache_dir.parent.mkdir(parents=True,exist_ok=True)
            if cache_dir.exists(): shutil.rmtree(cache_dir)
            shutil.move(str(generated/'preprocess'),str(cache_dir))
            for name in ['smpl_params.pt','used_bbx_xys.pt','1_incam.mp4']:
                shutil.move(str(generated/name),str(out/name))
            shutil.copy2(out/'1_incam.mp4',out/'preview_reprojection.mp4')
            shutil.rmtree(demo_root)
            run([sys.executable,'scripts/idd_ped/export_targetlocked.py','--clip_id',cid,
                 '--target_video',str(target_video),'--output_dir',str(out),'--preprocess_dir',str(cache_dir)])
            required=['smpl_params.pt','camera_trajectory.npz','ego_trajectory_unity.json','human_trajectory_unity.json',
                      'interaction_labels.json','preview_reprojection.mp4','preview_global.mp4','quality_report.json',
                      'target_track.npz','target_bbox_overlay.mp4','used_bbx_xys.pt']
            missing=[x for x in required if not (out/x).is_file() or (out/x).stat().st_size==0]
            if missing: raise RuntimeError(f'Missing outputs: {missing}')
            summary.append({'clip_id':cid,'status':'ok','output':str(out)})
        except Exception as exc:
            summary.append({'clip_id':cid,'status':'failed','error':str(exc),'output':str(out)})
            (args.output_root/'run_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); raise
        (args.output_root/'run_summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__': main()
