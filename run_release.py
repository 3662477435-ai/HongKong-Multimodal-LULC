"""One-command verification, archived-metric reproduction and frozen-model evaluation."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def models():
    return read_json(ROOT/"main_models.json")


def digest(path):
    out=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):out.update(chunk)
    return out.hexdigest()


def verify():
    failures=[];count=0
    for line in (ROOT/"checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected,relative=line.split("  ",1)
        path=ROOT/relative
        count+=1
        if not path.is_file() or digest(path)!=expected:failures.append(relative)
    print(json.dumps({"files_checked":count,"failures":failures},indent=2))
    return 1 if failures else 0


def metrics():
    import numpy as np
    results=[]
    for item in models():
        saved=read_json(ROOT/item["metrics"])["test"]["official8"]
        matrix=np.asarray(saved["confusion_matrix"],dtype=float)
        tp=np.diag(matrix);support=matrix.sum(1);predicted=matrix.sum(0)
        precision=np.divide(tp,predicted,out=np.full(8,np.nan),where=predicted>0)
        recall=np.divide(tp,support,out=np.full(8,np.nan),where=support>0)
        f1=np.divide(2*precision*recall,precision+recall,out=np.full(8,np.nan),where=precision+recall>0)
        iou=tp/(support+predicted-tp)
        oa=tp.sum()/matrix.sum();pe=np.dot(support,predicted)/matrix.sum()**2
        calculated={"overall_accuracy":float(oa),"macro_iou_present":float(np.nanmean(iou)),"macro_f1_present":float(np.nanmean(f1))}
        errors={key:abs(value-saved[key]) for key,value in calculated.items()}
        results.append({"model":item["name"],**calculated,"kappa":float((oa-pe)/(1-pe)),"valid_pixels":int(matrix.sum()),"maximum_difference":max(errors.values())})
    output=ROOT/"reproduced_metrics.json"
    output.write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(json.dumps({"models":len(results),"max_difference":max(x["maximum_difference"] for x in results),"output":str(output)},indent=2))
    return 1 if any(x["maximum_difference"]>1e-10 for x in results) else 0


def load_config(path):
    import yaml
    path=Path(path).resolve()
    config=yaml.safe_load(path.read_text(encoding="utf-8"))
    for key,value in config["paths"].items():
        value=Path(value)
        config["paths"][key]=str(value if value.is_absolute() else path.parent/value)
    config["data"].update(manifest_name="manifest_hier_v2.csv",vocab_name="vocab_hier_v2.json",normalization_name="normalization_hier_v2.json")
    return config


def preflight(config_path):
    config=load_config(config_path)
    inputs=[config["paths"][key] for key in ("image","osm_poi_raster","osm_aoi_raster","highres_aligned","nighttime_aligned","building_aligned","roads_aligned")]
    prepared=Path(config["paths"]["prepared_dir"])
    inputs.extend(str(prepared/name) for name in ("labels_leaf14.tif","manifest_hier_v2.csv","vocab_hier_v2.json","normalization_hier_v2.json"))
    inputs.extend(str(ROOT/item["checkpoint"]) for item in models())
    missing=[path for path in inputs if not Path(path).is_file()]
    print(json.dumps({"required_files":len(inputs),"missing":missing,"note":"See DATA_ACCESS.md for exact third-party inputs. No synthetic replacement is used for evaluation."},indent=2))
    return missing


def make_model(item,config,channels):
    import torch
    from hk_landuse.hierarchical_fcnn_model import HierarchicalMultiModalFCNN
    from hk_landuse.comparison_models import MultiModalEarlyFusionComparator
    common=dict(vocab_size=channels["vocab"],s2_channels=channels["s2"],s1_channels=channels["s1"],urban_channels=channels["urban"],road_channels=channels["roads"],osm_channels=20,text_dim=config["model"]["text_dim"],text_heads=config["model"]["text_heads"],text_layers=config["model"]["text_layers"],max_tokens=config["data"]["max_text_tokens"],dropout=config["model"].get("dropout",0.1),use_text=True)
    hierarchical=item["backbone"]=="ours"
    model=HierarchicalMultiModalFCNN(**common) if hierarchical else MultiModalEarlyFusionComparator(backbone=item["backbone"],num_classes=8,pretrained=False,**common)
    checkpoint=torch.load(ROOT/item["checkpoint"],map_location="cpu",weights_only=True)
    model.load_state_dict(checkpoint["model_state"],strict=True)
    return model.eval(),hierarchical


def smoke(config_path):
    import torch
    config=load_config(config_path)
    vocab=read_json(Path(config["paths"]["prepared_dir"])/"vocab_hier_v2.json")
    channels=dict(vocab=len(vocab),s2=19,s1=7,urban=20,roads=13)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results=[]
    # This is intentionally only a structural smoke check with synthetic
    # inputs; it cannot be mistaken for a held-out accuracy measurement.
    for item in models():
        model,hierarchical=make_model(item,config,channels)
        model=model.to(device)
        args=[torch.zeros(1,n,128,128,device=device) for n in (19,7,20,13,20)]
        ids=torch.ones(1,96,dtype=torch.long,device=device)
        mask=torch.ones(1,96,dtype=torch.bool,device=device)
        with torch.inference_mode():
            result=model(*args,ids,mask)
            logits=result["combined_log_probs"] if hierarchical else result
        finite=bool(torch.isfinite(logits).all())
        results.append({"model":item["name"],"strict_state_load":True,"shape":list(logits.shape),"finite":finite})
        del model,result,logits
        if device.type=="cuda":torch.cuda.empty_cache()
    (ROOT/"smoke_check.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(json.dumps({"models":len(results),"device":str(device),"all_finite":all(x["finite"] for x in results),"kind":"synthetic forward only; no accuracy claim"},indent=2))
    return 0 if all(x["finite"] and x["shape"]==[1,8,128,128] for x in results) else 1


def evaluate(config_path,selection):
    if preflight(config_path):return 2
    import torch
    from torch.utils.data import DataLoader
    from hk_landuse.train_comparison import make_dataset
    from hk_landuse.metrics import ConfusionMetrics
    from hk_landuse.task8 import CLASS_NAMES_8,leaf_target_to_official8
    config=load_config(config_path)
    modality_set={"s2","s1","highres","nighttime","building","roads","poi","aoi","text"}
    dataset=make_dataset(config,"test",False,modality_set)
    channels=dict(vocab=len(dataset.vocab),s2=dataset.s2_output_channels,s1=dataset.s1_output_channels,urban=dataset.urban_channels,roads=dataset.road_channels)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output=ROOT/"reproduced_test";output.mkdir(exist_ok=True)
    for item in models():
        if selection!="all" and selection!=item["id"]:continue
        model,hierarchical=make_model(item,config,channels)
        model=model.to(device)
        metric=ConfusionMetrics(8,CLASS_NAMES_8)
        with torch.inference_mode():
            for batch in DataLoader(dataset,batch_size=2,shuffle=False,num_workers=0):
                target=leaf_target_to_official8(batch["label"].to(device))
                with torch.amp.autocast(device_type=device.type,enabled=device.type=="cuda",dtype=torch.bfloat16):
                    result=model(*[batch[key].to(device) for key in ("s2","s1","urban","roads","osm","input_ids","text_mask")])
                    logits=result["combined_log_probs"] if hierarchical else result
                metric.update(logits,target)
        record={"model":item["name"],"device":str(device),"test":{"official8":metric.compute()}}
        name=item["id"].replace("/","__")+".json"
        (output/name).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps({"model":item["name"],"OA":record["test"]["official8"]["overall_accuracy"]}),flush=True)
        del model
        if device.type=="cuda":torch.cuda.empty_cache()
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("verify","metrics","preflight","smoke","evaluate"))
    parser.add_argument("--config",type=Path,default=ROOT/"config_release.yaml")
    parser.add_argument("--model",default="all",help="Exact id from main_models.json, or all")
    args=parser.parse_args()
    if args.command=="verify":code=verify()
    elif args.command=="metrics":code=metrics()
    elif args.command=="preflight":code=2 if preflight(args.config) else 0
    elif args.command=="smoke":code=smoke(args.config)
    else:code=evaluate(args.config,args.model)
    raise SystemExit(code)


if __name__=="__main__":main()
