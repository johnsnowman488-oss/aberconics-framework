"""Validation-only G2 readout diagnostic. Opens train/valid only; never opens test."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, random, statistics
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from ..digital.babi import (BabiDataset, babi_story_to_episode, build_babi_answer_set,
    build_babi_bridge, build_babi_memory_config, build_babi_vocabulary)
from ..digital.bridge import seed_memory_state, step_memory
from ..digital.readout import LinearReadout, query_conditioned_features

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DATA_DIR = _REPO_ROOT / "code" / "python" / "d2c" / "progress" / "babi_data"
CANDIDATES = (
    {"feature_scale": 100.0, "weight_decay": 0.0, "label": "g2_baseline"},
    {"feature_scale": 30.0, "weight_decay": 0.0, "label": "scale_30"},
    {"feature_scale": 10.0, "weight_decay": 0.0, "label": "scale_10"},
    {"feature_scale": 10.0, "weight_decay": 1e-4, "label": "scale_10_l2_1e-4"},
)

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def softmax(logits):
    z=logits-np.max(logits,axis=1,keepdims=True)
    e=np.exp(z)
    return e/np.sum(e,axis=1,keepdims=True)

def ece(conf, correct, bins=15):
    edges=np.linspace(0.0,1.0,bins+1); value=0.0
    for i in range(bins):
        mask=(conf>=edges[i]) & ((conf<=edges[i+1]) if i==bins-1 else (conf<edges[i+1]))
        if np.any(mask):
            value += float(np.mean(mask))*abs(float(np.mean(conf[mask]))-float(np.mean(correct[mask])))
    return value

def evaluate(readout, x, y):
    logits=np.asarray([readout.W@row+readout.b for row in x])
    probs=softmax(logits)
    pred=np.argmax(probs,axis=1); conf=np.max(probs,axis=1)
    correct=(pred==y).astype(float)
    onehot=np.eye(probs.shape[1])[y]
    high=conf>=0.90
    return {
      "accuracy":float(np.mean(correct)),
      "nll":float(-np.mean(np.log(np.maximum(probs[np.arange(len(y)),y],1e-12)))),
      "brier":float(np.mean(np.sum((probs-onehot)**2,axis=1))),
      "ece_15":ece(conf,correct),
      "mean_confidence":float(np.mean(conf)),
      "mean_confidence_on_errors":float(np.mean(conf[correct==0])) if np.any(correct==0) else None,
      "high_confidence_error_fraction_90":float(np.mean(correct[high]==0)) if np.any(high) else None,
      "high_confidence_coverage_90":float(np.mean(high)),
      "logit_abs_max_mean":float(np.mean(np.max(np.abs(logits),axis=1))),
      "logit_abs_max_p95":float(np.quantile(np.max(np.abs(logits),axis=1),0.95)),
    }

def regularized_step(model,x,target,lr,decay):
    logits=model.W@x+model.b
    z=logits-np.max(logits); ex=np.exp(z); p=ex/np.sum(ex)
    loss=-math.log(max(float(p[target]),1e-12))
    grad=p.copy(); grad[target]-=1.0
    model.W-=lr*(np.outer(grad,x)+decay*model.W)
    model.b-=lr*grad
    return float(loss)

def prepare(task,data_dir,dim,scales):
    train=BabiDataset.from_jsonl(data_dir/"babi_train.jsonl").filter_task(task)
    valid=BabiDataset.from_jsonl(data_dir/"babi_valid.jsonl").filter_task(task)
    answers=build_babi_answer_set(train.stories)
    vocab=build_babi_vocabulary(train.stories+valid.stories)
    bridge=build_babi_bridge(vocab,dim=dim)
    memory=build_babi_memory_config(variant="full")
    def encode(stories):
        states=[]; targets=[]
        for story in stories:
            ep=babi_story_to_episode(story,bridge,answers)
            state=seed_memory_state(state_dim=dim,channel_count=memory.channel_count)
            for forcing in ep.forcings: state=step_memory(state,forcing,memory)
            states.append((state,ep.query)); targets.append(answers.index(ep.target))
        matrices={scale:np.asarray([
            query_conditioned_features(state,memory,query,feature_scale=scale)
            for state,query in states
        ],dtype=np.float64) for scale in scales}
        return matrices,np.asarray(targets,dtype=np.int64)
    tx,ty=encode(train.stories); vx,vy=encode(valid.stories)
    return {"tx":tx,"ty":ty,"vx":vx,"vy":vy,"train_n":len(train.stories),
            "valid_n":len(valid.stories),"answers":answers}

def run(data_dir,output_dir,tasks=(3,5),seeds=5,epochs=5,dim=64,lr=0.05):
    data_dir=Path(data_dir); output_dir=Path(output_dir)
    train_path=data_dir/"babi_train.jsonl"; valid_path=data_dir/"babi_valid.jsonl"
    if not train_path.is_file() or not valid_path.is_file():
        raise FileNotFoundError("train and valid splits are required")
    scales=sorted({float(c["feature_scale"]) for c in CANDIDATES})
    prepared={t:prepare(t,data_dir,dim,scales) for t in tasks}
    tasks_out={}
    for task,data in prepared.items():
        candidate_rows=[]
        for cand in CANDIDATES:
            scale=float(cand["feature_scale"]); decay=float(cand["weight_decay"])
            xt,yt=data["tx"][scale],data["ty"]; xv,yv=data["vx"][scale],data["vy"]
            seed_rows=[]
            for seed in range(seeds):
                model=LinearReadout(input_dim=xt.shape[1],output_dim=len(data["answers"]),seed=seed)
                rng=random.Random(seed); epochs_out=[]
                for epoch in range(epochs):
                    order=list(range(len(yt))); rng.shuffle(order); losses=[]
                    for ix in order:
                        if decay==0.0:
                            loss=model.train_step(xt[ix],int(yt[ix]),learning_rate=lr)
                        else:
                            loss=regularized_step(model,xt[ix],int(yt[ix]),lr,decay)
                        losses.append(float(loss))
                    epochs_out.append({"epoch":epoch+1,"mean_train_loss":float(np.mean(losses)),
                                       **evaluate(model,xv,yv)})
                seed_rows.append({
                    "seed":seed,
                    "feature_norm_p50":float(np.quantile(np.linalg.norm(xt,axis=1),0.50)),
                    "feature_norm_p95":float(np.quantile(np.linalg.norm(xt,axis=1),0.95)),
                    "epochs":epochs_out,"final":epochs_out[-1]})
            epoch_summaries=[]
            for epoch_index in range(epochs):
                epoch_rows=[seed_row["epochs"][epoch_index] for seed_row in seed_rows]
                epoch_summaries.append({
                    "epoch":epoch_index+1,
                    "mean_validation_accuracy":statistics.mean(x["accuracy"] for x in epoch_rows),
                    "mean_validation_nll":statistics.mean(x["nll"] for x in epoch_rows),
                    "mean_validation_brier":statistics.mean(x["brier"] for x in epoch_rows),
                    "mean_validation_ece_15":statistics.mean(x["ece_15"] for x in epoch_rows),
                    "mean_confidence":statistics.mean(x["mean_confidence"] for x in epoch_rows),
                    "mean_logit_abs_max_p95":statistics.mean(x["logit_abs_max_p95"] for x in epoch_rows)})
            candidate_rows.append({
                **cand,"seeds":seed_rows,"epoch_summaries":epoch_summaries,
                "mean_validation_accuracy":statistics.mean(s["final"]["accuracy"] for s in seed_rows),
                "mean_validation_nll":statistics.mean(s["final"]["nll"] for s in seed_rows),
                "mean_validation_brier":statistics.mean(s["final"]["brier"] for s in seed_rows),
                "mean_validation_ece_15":statistics.mean(s["final"]["ece_15"] for s in seed_rows),
                "mean_confidence":statistics.mean(s["final"]["mean_confidence"] for s in seed_rows),
                "mean_logit_abs_max_p95":statistics.mean(s["final"]["logit_abs_max_p95"] for s in seed_rows)})
        chosen=min(candidate_rows,key=lambda r:(r["mean_validation_nll"],-r["mean_validation_accuracy"]))
        checkpoint_options=[(candidate,epoch_row) for candidate in candidate_rows for epoch_row in candidate["epoch_summaries"]]
        best_candidate,best_epoch=min(checkpoint_options,key=lambda item:(item[1]["mean_validation_nll"],-item[1]["mean_validation_accuracy"]))
        tasks_out[str(task)]={"task":task,"train_stories":data["train_n"],
          "validation_stories":data["valid_n"],"answer_count":len(data["answers"]),
          "candidates":candidate_rows,
          "selected_by_validation_only":{"label":chosen["label"],"feature_scale":chosen["feature_scale"],
            "weight_decay":chosen["weight_decay"],
            "selection_rule":"lowest mean validation NLL at final configured epoch; validation accuracy tie-break"},
          "selected_validation_checkpoint":{"label":best_candidate["label"],
            "feature_scale":best_candidate["feature_scale"],"weight_decay":best_candidate["weight_decay"],
            "epoch":best_epoch["epoch"],"metrics":best_epoch,
            "selection_rule":"lowest mean validation NLL across all candidate/epoch checkpoints; validation accuracy tie-break"}}
    result={
      "experiment":"g2_validation_only_readout_check","status":"completed",
      "created_at":datetime.now(timezone.utc).isoformat(),"evaluation_split":"valid",
      "held_out_test_split_opened":False,
      "claim_boundary":"Exploratory readout scale/regularization selection on train and validation only; not a test result.",
      "config":{"tasks":list(tasks),"seeds":seeds,"epochs":epochs,"dim":dim,"learning_rate":lr,
                "candidates":list(CANDIDATES)},
      "dataset_fingerprints":{"train_sha256":sha256(train_path),"valid_sha256":sha256(valid_path)},
      "tasks":tasks_out}
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    lab_register={
      "experiment_id":"G2-VALIDATION-READOUT",
      "experiment":result["experiment"],
      "status":"completed",
      "run_timestamp":result["created_at"],
      "evaluation_split":"valid",
      "held_out_test_split_opened":False,
      "claim_boundary":result["claim_boundary"],
      "config":result["config"],
      "dataset_fingerprints":result["dataset_fingerprints"],
      "task_results":{
        task:{"selected_final_epoch":row["selected_by_validation_only"],
              "selected_validation_checkpoint":row["selected_validation_checkpoint"]}
        for task,row in result["tasks"].items()},
      "artifacts":["summary.json","summary.md","per_seed_metrics.csv","experiment_lab_register.json"]
    }
    (output_dir/"experiment_lab_register.json").write_text(json.dumps(lab_register,indent=2),encoding="utf-8")
    fields=["task","candidate","feature_scale","weight_decay","seed","epoch","train_loss",
      "validation_accuracy","validation_nll","validation_brier","validation_ece_15",
      "mean_confidence","logit_abs_max_p95","feature_norm_p50","feature_norm_p95"]
    with (output_dir/"per_seed_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for task_key,tr in tasks_out.items():
            for cand in tr["candidates"]:
                for sr in cand["seeds"]:
                    for ep in sr["epochs"]:
                        writer.writerow({"task":task_key,"candidate":cand["label"],
                          "feature_scale":cand["feature_scale"],"weight_decay":cand["weight_decay"],
                          "seed":sr["seed"],"epoch":ep["epoch"],"train_loss":ep["mean_train_loss"],
                          "validation_accuracy":ep["accuracy"],"validation_nll":ep["nll"],
                          "validation_brier":ep["brier"],"validation_ece_15":ep["ece_15"],
                          "mean_confidence":ep["mean_confidence"],
                          "logit_abs_max_p95":ep["logit_abs_max_p95"],
                          "feature_norm_p50":sr["feature_norm_p50"],"feature_norm_p95":sr["feature_norm_p95"]})
    lines=["# G2 validation-only readout check","","Only train and validation splits were opened; held-out test was not opened.","",
      "| Task | Candidate | Scale | L2 | Val acc | Val NLL | Brier | ECE | Mean conf. | p95 max logit |",
      "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for task_key,tr in tasks_out.items():
        for r in tr["candidates"]:
            lines.append(f"| QA{task_key} | {r['label']} | {r['feature_scale']:g} | {r['weight_decay']:g} | "
              f"{r['mean_validation_accuracy']:.4f} | {r['mean_validation_nll']:.4f} | "
              f"{r['mean_validation_brier']:.4f} | {r['mean_validation_ece_15']:.4f} | "
              f"{r['mean_confidence']:.4f} | {r['mean_logit_abs_max_p95']:.3f} |")
    lines += ["","## Validation-selected settings",""]
    for task_key,tr in tasks_out.items():
        c=tr["selected_by_validation_only"]
        lines.append(f"- QA{task_key}: {c['label']} (scale={c['feature_scale']}, L2={c['weight_decay']}).")
    (output_dir/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,default=_DATA_DIR)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--tasks",default="3,5"); p.add_argument("--seeds",type=int,default=5)
    p.add_argument("--epochs",type=int,default=5); p.add_argument("--dim",type=int,default=64)
    p.add_argument("--learning-rate",type=float,default=0.05)
    a=p.parse_args(); tasks=tuple(int(v) for v in a.tasks.split(","))
    if not tasks or any(t not in (3,5) for t in tasks) or len(set(tasks))!=len(tasks):
        p.error("--tasks must be 3, 5, or 3,5")
    if a.seeds<3 or a.epochs<1: p.error("use at least 3 seeds and 1 epoch")
    result=run(a.data_dir,a.output_dir,tasks,a.seeds,a.epochs,a.dim,a.learning_rate)
    for task,tr in result["tasks"].items():
        print(f"QA{task} selected={tr['selected_by_validation_only']}")
        for c in tr["candidates"]:
            print(f"  {c['label']}: val_acc={c['mean_validation_accuracy']:.4f} "
              f"NLL={c['mean_validation_nll']:.4f} ECE={c['mean_validation_ece_15']:.4f} "
              f"logit_p95={c['mean_logit_abs_max_p95']:.3f}")
    print(f"validation-only evidence: {a.output_dir}")

if __name__=="__main__": main()
