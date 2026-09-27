"""Download a seeded random subset of Google Scanned Objects (CC BY 4.0) from Gazebo Fuel."""
import json, os, random, subprocess, sys, time, urllib.request
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "gso"); N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
os.makedirs(f"{OUT}/zips", exist_ok=True)
lst = f"{OUT}/all_models.json"
if not os.path.exists(lst):
    allm, page = [], 1
    while True:
        u = f"https://fuel.gazebosim.org/1.0/models?per_page=100&page={page}&q=owner:GoogleResearch"
        try:
            d = json.load(urllib.request.urlopen(u, timeout=60))
        except Exception as e:
            print("list error", page, e); break
        if not d: break
        allm += [dict(name=m["name"], license=m["license_name"], categories=m.get("categories"), filesize=m["filesize"]) for m in d if m["owner"] == "GoogleResearch"]
        page += 1; time.sleep(0.5)
    json.dump(allm, open(lst, "w"), indent=0)
allm = json.load(open(lst)); print(len(allm), "models listed")
assert all(m["license"] == "Creative Commons Attribution 4.0 International" for m in allm)
rng = random.Random(0)
names = sorted(m["name"] for m in allm)
pick = rng.sample(names, N)
json.dump(pick, open(f"{OUT}/subset_seed0.json", "w"), indent=0)
for i, n in enumerate(pick):
    z = f"{OUT}/zips/{n}.zip"
    if os.path.exists(z) and os.path.getsize(z) > 0: continue
    for t in range(4):
        r = subprocess.run(["curl", "-sfL", "-m", "300", "-o", z + ".part", f"https://fuel.gazebosim.org/1.0/GoogleResearch/models/{n}.zip"])
        if r.returncode == 0: os.replace(z + ".part", z); break
        time.sleep(5 * (t + 1))
    print(i, n, "ok" if os.path.exists(z) else "FAILED", flush=True)
print("ALLDONE")
