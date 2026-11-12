"""Validation-only LR selection then three-seed direct-48 strong baselines."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
PREFIX="eight_class_lineage_strong_baselines_20260813"
MODELS=("segformer_b1_direct48","deeplabv3plus_resnet50_direct48","swin_t_upernet_direct48")
LRS=("3e-5","6e-5","1e-4")
MODALITIES="s2,s1,highres,nighttime,building,roads,poi,aoi,text"
def call(cmd,log):
 log.write("COMMAND "+" ".join(cmd)+"\n");log.flush()
 # Keep long patch-wise raster reads within a bounded GDAL cache; this avoids
 # transient file-backed cache allocation failures on the system drive.
 p=subprocess.Popen(cmd,cwd=REPO,env={**os.environ,"PYTHONPATH":str(REPO/'src'),"GDAL_CACHEMAX":"128"},stdout=log,stderr=subprocess.STDOUT,text=True)
 if p.wait(): raise RuntimeError(cmd)
def score(path): return json.loads(path.read_text(encoding="utf8"))["validation"]["official8"]["macro_iou_present"]
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--python',default=sys.executable);a=p.parse_args()
 root=REPO/'outputs/full'/PREFIX; logs=root/'logs';logs.mkdir(parents=True,exist_ok=True)
 with (logs/f'{a.model}.log').open('a',encoding='utf8') as log:
  candidates=[]
  for lr in LRS:
   name=f'{a.model}_lr{lr}_seed42'; out=root/name
   if not (out/'metrics.json').exists(): call([a.python,'-u','-m','hk_landuse.train_comparison_8class','--config','config_release.yaml','--model',a.model,'--run-name',f'{PREFIX}/{name}','--epochs','47','--patience','0','--batch-size','4','--num-workers','0','--seed','42','--modalities',MODALITIES,'--backbone-learning-rate',lr,'--manifest-name','manifest_hier_v2.csv','--vocab-name','vocab_hier_v2.json','--normalization-name','normalization_hier_v2.json'],log)
   candidates.append((score(out/'metrics.json'),lr,out/'best_model.pt'))
  _, best_lr, best_ckpt=max(candidates)
  for seed in (43,44):
   name=f'{a.model}_lr{best_lr}_seed{seed}';out=root/name
   if not (out/'metrics.json').exists(): call([a.python,'-u','-m','hk_landuse.train_comparison_8class','--config','config_release.yaml','--model',a.model,'--run-name',f'{PREFIX}/{name}','--epochs','47','--patience','0','--batch-size','4','--num-workers','0','--seed',str(seed),'--modalities',MODALITIES,'--backbone-learning-rate',best_lr,'--manifest-name','manifest_hier_v2.csv','--vocab-name','vocab_hier_v2.json','--normalization-name','normalization_hier_v2.json'],log)
  (root/f'{a.model}_selection.json').write_text(json.dumps({'selected_backbone_lr':best_lr,'seed42_checkpoint':str(best_ckpt),'validation_candidates':[(x,y) for x,y,_ in candidates]},indent=2),encoding='utf8')
if __name__=='__main__': main()
