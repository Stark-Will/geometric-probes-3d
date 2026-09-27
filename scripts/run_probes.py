"""Memory-safe batch runner: one mesh per subprocess, wall-clock timeout, RSS watchdog.

Usage: python scripts/run_probes.py --glob 'data/3d-defectbench/glb/*/*.glb' --out results/probes/defectbench
Writes one JSON per mesh (skips existing -> resumable) and aggregates to <out>.csv.
"""
import argparse, glob, json, os, subprocess, sys, time
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORKER = r'''
import json, sys, time, traceback, platform
sys.path.insert(0, sys.argv[3])
t0 = time.time()
try:
    from geomcheck.probes import compute_all, CONFIG
    import trimesh, pymeshlab, embreex, numpy
    dec = None if sys.argv[4] == "none" else int(sys.argv[4])
    r = compute_all(sys.argv[1], decimate_to=dec); r["status"] = "ok"
except Exception as e:
    r = {"status": "error", "error": repr(e), "trace": traceback.format_exc()[-2000:]}
r["path"] = sys.argv[1]; r["seconds"] = round(time.time() - t0, 3)
json.dump(r, open(sys.argv[2], "w"))
'''


def run_one(path, out_json, timeout, rss_limit, dec="none"):
    p = subprocess.Popen([sys.executable, "-c", WORKER, path, out_json, ROOT, str(dec)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                         env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    proc = psutil.Process(p.pid)
    t0, peak, reason = time.time(), 0, None
    while p.poll() is None:
        try:
            rss = proc.memory_info().rss
            peak = max(peak, rss)
        except psutil.Error:
            break
        if rss > rss_limit:
            reason = f"killed_rss>{rss_limit/2**30:.1f}GB"
        elif time.time() - t0 > timeout:
            reason = f"killed_timeout>{timeout}s"
        if reason:
            p.kill(); p.wait(); break
        time.sleep(0.2)
    if reason or not os.path.exists(out_json):
        json.dump({"path": path, "status": reason or "crashed",
                   "stderr": (p.stderr.read() or b"")[-1000:].decode(errors="replace"),
                   "seconds": round(time.time() - t0, 3)}, open(out_json, "w"))
    r = json.load(open(out_json)); r["peak_rss_mb"] = round(peak / 2**20, 1)
    json.dump(r, open(out_json, "w"))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--rss_gb", type=float, default=2.5)
    ap.add_argument("--decimate", default="none", help="target face count or 'none'")
    ap.add_argument("--strip", default=None, help="name JSON by path relative to this prefix")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    paths = sorted(glob.glob(a.glob))
    todo = []
    for pth in paths:
        if a.strip:
            tag = os.path.splitext(os.path.relpath(pth, a.strip))[0].replace("/", "__")
        else:
            tag = pth.replace("/", "__").replace(".glb", "").split("glb__")[-1]
        oj = os.path.join(a.out, tag + ".json")
        if not os.path.exists(oj):
            todo.append((pth, oj))
    print(f"{len(paths)} meshes, {len(todo)} to do", flush=True)
    from concurrent.futures import ThreadPoolExecutor
    done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(lambda x: run_one(x[0], x[1], a.timeout, a.rss_gb * 2**30, a.decimate), todo):
            done += 1
            print(f"[{done}/{len(todo)}] {r['path']} {r['status']} {r.get('seconds')}s {r.get('peak_rss_mb')}MB", flush=True)
    import pandas as pd
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.out, "*.json")))]
    df = pd.DataFrame(rows).drop(columns=["trace"], errors="ignore")
    df.to_csv(a.out.rstrip("/") + ".csv", index=False)
    print("status counts:", df["status"].value_counts().to_dict())


if __name__ == "__main__":
    main()
