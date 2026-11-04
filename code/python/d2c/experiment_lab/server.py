"""Local-first Experiment Lab API. Run with: python -m d2c.experiment_lab.server"""
from __future__ import annotations
import hashlib, json, math, os, platform, re, subprocess, sys, threading, uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[4]
RUNS = ROOT / "runs" / "experiment-lab"
DATA = ROOT / "code/python/d2c/progress/babi_data"
LOCK = threading.Lock()
ACTIVE = None
JOBS = {}

def utc(): return datetime.now(timezone.utc).isoformat()
def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
def code_revision():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"
def write_manifest(job):
    directory = Path(job["directory"])
    payload = public_job(job)
    payload.update({
        "manifest_version": 1,
        "command": job.get("command", []),
        "code_revision": job.get("code_revision", "unknown"),
        "python_version": platform.python_version(),
        "dataset_fingerprints": job.get("dataset_fingerprints", {}),
        "result_files": sorted(
            p.relative_to(directory).as_posix()
            for p in directory.rglob("*")
            if p.is_file() and p.name not in {"metadata.json", "manifest.json"}
        ),
    })
    (directory / "metadata.json").write_text(json.dumps(payload, indent=2))
    (directory / "manifest.json").write_text(json.dumps(payload, indent=2))
def load_history():
    RUNS.mkdir(parents=True, exist_ok=True)
    for p in RUNS.glob("*/metadata.json"):
        try: JOBS[p.parent.name] = json.loads(p.read_text())
        except Exception: pass
def public_job(job): return {k:v for k,v in job.items() if k not in {"process","command"}}

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def send(self, status, obj, content_type="application/json"):
        body = obj if isinstance(obj, bytes) else (json.dumps(obj).encode() if content_type=="application/json" else str(obj).encode())
        self.send_response(status); self.send_header("Content-Type", content_type+"; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control","no-store"); self.end_headers(); self.wfile.write(body)
    def body(self):
        n=int(self.headers.get("Content-Length","0"))
        if n>16384: raise ValueError("request too large")
        return json.loads(self.rfile.read(n) or b"{}")
    def do_GET(self):
        path=urlparse(self.path).path
        if path=="/": return self.send(200,(ROOT/"code/python/d2c/experiment_lab/index.html").read_bytes(),"text/html")
        if path=="/api/experiments": return self.send(200,{"experiments":[{"id":"e0","name":"E0 bAbI baseline","modes":["isolated"],"learned":true},{"id":"ef0","name":"E-F0 runtime-integrated bAbI","modes":["isolated","continuous"],"learned":true},{"id":"g1","name":"G1 explicit relational-state QA2 diagnostic","modes":["isolated"],"task":2,"learned":false},{"id":"g1l","name":"G1-L learned relational binding","modes":["isolated"],"task":2,"learned":true,"stages":["L0","L1","L2"]},{"id":"g2","name":"G2 relational composition QA3/QA5","modes":["isolated"],"tasks":[3,5],"learned":true,"arms":["e0","ef0"],"controls":["structured_relational_fixtures"]}]})
        if path=="/api/datasets":
            files={s:(DATA/f"babi_{s}.jsonl").exists() for s in ("train","test","valid")}
            return self.send(200,{"files":files,"ready":files["train"] and files["test"],"ready_for_g1l":all(files.values()),"ready_for_g2":files["train"] and files["test"]})
        if path=="/api/runs": return self.send(200,{"runs":[public_job(x) for x in sorted(JOBS.values(),key=lambda j:j["created_at"],reverse=True)]})
        if path=="/api/compare":
            ids=parse_qs(urlparse(self.path).query).get("ids",[""])[0].split(",")
            ids=[rid for rid in ids if rid]
            if len(ids)<2 or len(ids)>5:
                return self.send(400,{"error":"select between two and five runs to compare"})
            rows=[]
            for rid in ids:
                job=JOBS.get(rid)
                if not job: return self.send(404,{"error":f"run not found: {rid}"})
                summaries=list(Path(job["directory"]).glob("**/summary.json"))
                result=json.loads(summaries[0].read_text()) if summaries else {}
                stages = result.get("stages", {})
                l1 = stages.get("L1", {})
                l2 = stages.get("L2", {})
                l1_validation = l1.get("validation_after_training", {})
                rows.append({
                    "run_id":rid,"experiment":job.get("experiment"),
                    "status":job.get("status"),"created_at":job.get("created_at"),
                    "config":job.get("config"),
                    "mean_accuracy":(
                        result.get("mean_accuracy")
                        if result.get("mean_accuracy") is not None
                        else (
                            result.get("stages",{}).get("L2",{}).get("mean_accuracy")
                            if result.get("stages",{}).get("L2",{}).get("status")=="completed"
                            else result.get("stages",{}).get("L1",{}).get("validation_after_training",{}).get("accuracy")
                        )
                    ),
                    "accuracy_split":(
                        "test" if result.get("stages",{}).get("L2",{}).get("status")=="completed"
                        else ("validation" if result.get("experiment")=="babi_g1l_learned_relational_binding" else result.get("evaluation_split","test"))
                    ),
                    "standard_error":result.get("standard_error",result.get("se_accuracy",result.get("stages",{}).get("L2",{}).get("standard_error"))),
                    "per_seed_accuracy":result.get("per_seed_accuracy",result.get("stages",{}).get("L2",{}).get("per_seed_accuracy")),
                    "l1_status":l1.get("status"),
                    "l2_status":l2.get("status"),
                    "held_query_accuracy":(
                        l2.get("mean_held_query_accuracy")
                        if l2.get("status") == "completed"
                        else l1_validation.get("held_query_accuracy")
                    ),
                    "holder_retrieval_accuracy":(
                        l2.get("mean_holder_retrieval_accuracy")
                        if l2.get("status") == "completed"
                        else l1_validation.get("holder_retrieval_accuracy")
                    ),
                    "fixed_location_accuracy":(
                        l2.get("mean_fixed_location_accuracy")
                        if l2.get("status") == "completed"
                        else l1_validation.get("fixed_location_accuracy")
                    ),
                    "dataset":result.get("dataset"),
                })
            return self.send(200,{"runs":rows})
        file_match=re.fullmatch(r"/api/runs/([a-f0-9-]+)/files/(summary\.json|summary\.md|per_example\.csv|per_example\.jsonl|manifest\.json|config\.json|experiment_lab_register\.json)",path)
        if file_match:
            rid,filename=file_match.groups(); job=JOBS.get(rid)
            if not job: return self.send(404,{"error":"run not found"})
            matches=[p for p in Path(job["directory"]).glob(f"**/{filename}") if p.is_file()]
            if not matches: return self.send(404,{"error":"result file not found"})
            types={"summary.json":"application/json","summary.md":"text/markdown","per_example.csv":"text/csv","per_example.jsonl":"application/x-ndjson","per_seed_metrics.csv":"text/csv","fixtures.jsonl":"application/x-ndjson","manifest.json":"application/json","config.json":"application/json","experiment_lab_register.json":"application/json"}
            return self.send(200,matches[0].read_bytes(),types[filename])
        m=re.fullmatch(r"/api/runs/([a-f0-9-]+)(?:/(results|logs))?",path)
        if m:
            job=JOBS.get(m.group(1))
            if not job: return self.send(404,{"error":"run not found"})
            if m.group(2)=="logs":
                p=Path(job["directory"])/"stdout.log"; q=Path(job["directory"])/"stderr.log"
                return self.send(200,{"stdout":p.read_text(errors="replace")[-12000:] if p.exists() else "","stderr":q.read_text(errors="replace")[-12000:] if q.exists() else ""})
            if m.group(2)=="results":
                matches=list(Path(job["directory"]).glob("**/summary.json"))
                return self.send(200,json.loads(matches[0].read_text()) if matches else {"pending":True})
            return self.send(200,public_job(job))
        return self.send(404,{"error":"not found"})
    def do_POST(self):
        global ACTIVE
        path=urlparse(self.path).path
        try: data=self.body()
        except Exception as e: return self.send(400,{"error":str(e)})
        if path=="/api/runs":
            with LOCK:
                if ACTIVE and ACTIVE.poll() is None: return self.send(409,{"error":"another experiment is running"})
                try:
                    exp=data.get("experiment"); mode=data.get("mode","isolated"); task=int(data.get("task",2)); dim=int(data.get("dim",64)); seeds=int(data.get("seeds",3)); epochs=int(data.get("epochs",5))
                    exercise_spine=data.get("exercise_spine",True); readout=data.get("readout","linear")
                    learning_rate=float(data.get("learning_rate",0.05)); feature_scale=float(data.get("feature_scale",100.0))
                    stage=data.get("stage","pilot")
                    train_limit=int(data.get("train_limit",200)); eval_limit=int(data.get("eval_limit",100))
                except (TypeError,ValueError): return self.send(400,{"error":"numeric settings are invalid"})
                if not isinstance(exercise_spine,bool):
                    return self.send(400,{"error":"exercise_spine must be boolean"})
                if (exp not in {"e0","ef0","g1","g1l","g2"} or mode not in {"isolated","continuous"}
                    or (exp in {"e0","g1","g1l"} and mode!="isolated")
                    or (exp in {"g1","g1l"} and task!=2)
                    or (exp=="g2" and (mode!="isolated" or seeds<3 or seeds>20 or dim<4 or not g2_tasks or any(value not in {3,5} for value in g2_tasks) or len(set(g2_tasks))!=len(g2_tasks)))
                    or (exp=="g1l" and (stage not in {"pilot","all"} or seeds>10 or dim<8))
                    or not (1<=train_limit<=900 and 1<=eval_limit<=1000)
                    or readout not in {"linear","mlp"}
                    or not math.isfinite(learning_rate) or learning_rate<=0
                    or not math.isfinite(feature_scale) or feature_scale<=0
                    or not (1<=task<=20 and 4<=dim<=512 and 1<=seeds<=100 and 1<=epochs<=100)):
                    return self.send(400,{"error":"invalid experiment configuration"})
                rid=str(uuid.uuid4()); directory=RUNS/rid; directory.mkdir(parents=True)
                module={"e0":"d2c.experiments.babi_qa","ef0":"d2c.experiments.babi_f0","g1":"d2c.experiments.babi_g1","g1l":"d2c.experiments.babi_g1l","g2":"d2c.experiments.babi_g2"}[exp]
                args=[sys.executable,"-m",module,"--dim",str(dim),"--output-dir",str(directory)]
                if exp!="g2": args += ["--task",str(task)]
                if exp not in {"g1l","g2"}: args += ["--variant","full"]
                if exp=="g1":
                    args += ["--trace-capacity","20000"]
                elif exp=="g1l":
                    args += ["--stage",stage,"--learning-rate",str(learning_rate),
                        "--epochs",str(epochs),"--train-limit",str(train_limit),
                        "--eval-limit",str(eval_limit),"--confirm-seeds",str(seeds)]
                elif exp=="g2":
                    args += ["--tasks",",".join(str(value) for value in g2_tasks),
                        "--seeds",str(seeds),"--epochs",str(epochs),
                        "--learning-rate",str(learning_rate),"--feature-scale",str(feature_scale),
                        "--readout",readout,"--bootstrap-samples","300","--eval-split","test"]
                else:
                    args += ["--seeds",str(seeds),"--readout",readout,"--lr",str(learning_rate),"--feature-scale",str(feature_scale),"--epochs",str(epochs),"--eval-split","test"]
                if exp=="ef0":
                    args += ["--mode",mode]
                    if exercise_spine: args.append("--exercise-spine")
                cfg={"experiment":exp,"task":None if exp=="g2" else task,"tasks":list(g2_tasks) if exp=="g2" else None,"dim":dim,"seeds":None if exp=="g1" else seeds,"epochs":None if exp=="g1" else epochs,"mode":"isolated" if exp=="g2" else mode,"variant":"full","readout":"relational_state_resolution" if exp=="g1" else ("learned_relational_binding" if exp=="g1l" else readout),"learning_rate":None if exp=="g1" else learning_rate,"feature_scale":None if exp=="g1" else feature_scale,"evaluation_split":"validation_then_test" if exp=="g1l" else "test","exercise_spine":bool(exercise_spine and exp=="ef0"),"f0_companion":exp=="g1l","learning_path_uses_f0_features":False if exp=="g1l" else None,"learning_performed":False if exp=="g1" else (True if exp in {"g1l","g2"} else None),"stage":stage if exp=="g1l" else None,"train_limit":train_limit if exp=="g1l" else None,"eval_limit":eval_limit if exp=="g1l" else None,"confidence_metrics":"full_probability+NLL+Brier+fixed/adaptive/classwise ECE+selective risk+story bootstrap" if exp=="g2" else None}
                (directory/"config.json").write_text(json.dumps(cfg,indent=2))
                dataset_fingerprints={}
                for split in ("train","valid","test"):
                    source=DATA/f"babi_{split}.jsonl"
                    if source.is_file():
                        dataset_fingerprints[split]={"sha256":file_sha256(source),"bytes":source.stat().st_size}
                job={"run_id":rid,"experiment":exp,"config":cfg,"status":"running","created_at":utc(),"started_at":utc(),"directory":str(directory),"command":args,"code_revision":code_revision(),"dataset_fingerprints":dataset_fingerprints}
                write_manifest(job)
                out=(directory/"stdout.log").open("w"); err=(directory/"stderr.log").open("w")
                env=os.environ.copy(); env["PYTHONPATH"]=str(ROOT/"code/python"); env["GFE_CORE_LIB"]=str(ROOT/"code/c++ core/build-shared/libgfe_core.so")
                try: proc=subprocess.Popen(args,cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True)
                except Exception as e:
                    out.close(); err.close(); return self.send(500,{"error":str(e)})
                ACTIVE=proc; job["process"]=proc; job["pid"]=proc.pid; JOBS[rid]=job
                def watch():
                    global ACTIVE
                    code=proc.wait(); out.close(); err.close()
                    job["status"]="completed" if code==0 else "failed"; job["exit_code"]=code; job["completed_at"]=utc(); job.pop("process",None)
                    write_manifest(job)
                    with LOCK:
                        if ACTIVE is proc: ACTIVE=None
                threading.Thread(target=watch,daemon=True).start()
                return self.send(202,public_job(job))
        if path.startswith("/api/runs/") and path.endswith("/cancel"):
            rid=path.split("/")[-2]; job=JOBS.get(rid)
            if not job or not job.get("process") or job["process"].poll() is not None: return self.send(409,{"error":"run is not active"})
            job["process"].terminate(); return self.send(202,{"status":"cancellation_requested"})
        return self.send(404,{"error":"not found"})

def main():
    load_history()
    print("Experiment Lab: http://127.0.0.1:8765")
    ThreadingHTTPServer(("127.0.0.1",8765),Handler).serve_forever()
if __name__=="__main__": main()
