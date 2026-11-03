"""Local-first Experiment Lab API. Run with: python -m d2c.experiment_lab.server"""
from __future__ import annotations
import json, os, re, subprocess, sys, threading, uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[4]
RUNS = ROOT / "runs" / "experiment-lab"
DATA = ROOT / "code/python/d2c/progress/babi_data"
LOCK = threading.Lock()
ACTIVE = None
JOBS = {}

def utc(): return datetime.now(timezone.utc).isoformat()
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
        if path=="/api/experiments": return self.send(200,{"experiments":[{"id":"e0","name":"E0 bAbI baseline","modes":["isolated"]},{"id":"ef0","name":"E-F0 runtime-integrated bAbI","modes":["isolated","continuous"]}]})
        if path=="/api/datasets":
            files={s:(DATA/f"babi_{s}.jsonl").exists() for s in ("train","test","valid")}
            return self.send(200,{"files":files,"ready":files["train"] and files["test"]})
        if path=="/api/runs": return self.send(200,{"runs":[public_job(x) for x in sorted(JOBS.values(),key=lambda j:j["created_at"],reverse=True)]})
        m=re.fullmatch(r"/api/runs/([a-f0-9-]+)(?:/(results|logs))?",path)
        if m:
            job=JOBS.get(m.group(1))
            if not job: return self.send(404,{"error":"run not found"})
            if m.group(2)=="logs":
                p=Path(job["directory"])/"stdout.log"; q=Path(job["directory"])/"stderr.log"
                return self.send(200,{"stdout":p.read_text(errors="replace")[-12000:] if p.exists() else "","stderr":q.read_text(errors="replace")[-12000:] if q.exists() else ""})
            if m.group(2)=="results":
                p=Path(job["directory"])/"summary.json"
                return self.send(200,json.loads(p.read_text()) if p.exists() else {"pending":True})
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
                    exp=data.get("experiment"); mode=data.get("mode","isolated"); task=int(data.get("task",1)); dim=int(data.get("dim",64)); seeds=int(data.get("seeds",3)); epochs=int(data.get("epochs",5))
                except (TypeError,ValueError): return self.send(400,{"error":"numeric settings are invalid"})
                if exp not in {"e0","ef0"} or mode not in {"isolated","continuous"} or (exp=="e0" and mode!="isolated") or not (1<=task<=20 and 4<=dim<=512 and 1<=seeds<=100 and 1<=epochs<=100):
                    return self.send(400,{"error":"invalid experiment configuration"})
                rid=str(uuid.uuid4()); directory=RUNS/rid; directory.mkdir(parents=True)
                module="d2c.experiments.babi_qa" if exp=="e0" else "d2c.experiments.babi_f0"
                args=[sys.executable,"-m",module,"--task",str(task),"--dim",str(dim),"--seeds",str(seeds),"--output-dir",str(directory),"--variant","full","--readout","linear"]
                if exp=="ef0": args += ["--epochs",str(epochs),"--mode",mode]
                cfg={"experiment":exp,"task":task,"dim":dim,"seeds":seeds,"epochs":epochs,"mode":mode,"variant":"full","readout":"linear"}
                (directory/"config.json").write_text(json.dumps(cfg,indent=2))
                job={"run_id":rid,"experiment":exp,"config":cfg,"status":"running","created_at":utc(),"directory":str(directory),"command":args}
                (directory/"metadata.json").write_text(json.dumps(public_job(job),indent=2))
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
                    (directory/"metadata.json").write_text(json.dumps(public_job(job),indent=2))
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
