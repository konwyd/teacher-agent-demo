from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from .baseline import DATA,LAYER_B,ROOT
MANIFEST=DATA/"benchmark_v1_manifest.json"
def sha256(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def tracked_paths()->dict[str,Path]:
    return {"development_dataset":DATA/"development.jsonl","sealed_holdout_dataset":DATA/"holdout.jsonl","evaluator":DATA.parent/"evaluator.py","models":DATA.parent/"models.py","baseline_adapter":DATA.parent/"baseline.py","layer_b_unchanged":LAYER_B,"memory_candidate_contract":ROOT/"app/memory/contracts.py"}
def verify_manifest(path:Path=MANIFEST)->list[str]:
    manifest=json.loads(path.read_text(encoding="utf-8")); failures=[]
    for name,file_path in tracked_paths().items():
        if sha256(file_path)!=manifest["runtime_verification"][name]["sha256"]:failures.append(name)
    return failures
def build_manifest()->dict:
    return {"benchmark_version":"candidate-extractor-v2-comparative-v1","historical_provenance":{"source_base_sha":"d7ed183f83c162b9dcbc78c5e7c992e758bf1497","created_at":datetime.now(timezone.utc).isoformat(),"routes_implemented":False},"label_seal":{"development_sha256":sha256(DATA/"development.jsonl"),"holdout_sha256":sha256(DATA/"holdout.jsonl"),"sealed_before_first_formal_baseline":True},"runtime_verification":{name:{"path":str(path.relative_to(ROOT)),"sha256":sha256(path)} for name,path in tracked_paths().items()}}
