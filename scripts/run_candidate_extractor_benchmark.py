#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.evaluation.candidate_extractor.baseline import DATA,run_baseline
from app.evaluation.candidate_extractor.manifest import MANIFEST,build_manifest,verify_manifest
def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=DATA/"baseline_current_extractor.json");parser.add_argument("--verify-manifest",action="store_true");parser.add_argument("--freeze-manifest",action="store_true");args=parser.parse_args()
    if args.verify_manifest:
        failures=verify_manifest()
        if failures:raise SystemExit("manifest drift: "+", ".join(failures))
    report=run_baseline();args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if args.freeze_manifest:MANIFEST.write_text(json.dumps(build_manifest(),ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    for split,result in report["reports"].items():print(split,result["case_count"],result["exact_candidate_set_accuracy"]["value"])
    return 0
if __name__=="__main__":raise SystemExit(main())
