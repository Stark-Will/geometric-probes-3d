"""Audit: every decimal/percentage/large integer in the LaTeX sources must match a value in a result file.
Pool = all numeric cells of results/**/*.csv, results/**/*.json, results/RESULTS_*.md, paper/figures/gallery_cases.csv,
paper/tables/*.tex, analysis/PREREGISTRATION*.md, geomcheck/probes.py CONFIG.
A token with d decimals matches if some pool value rounds to it at d decimals (sign-insensitive: LaTeX minus
signs are often typeset separately); a percentage p% with d decimals matches pool value p/100 at d+2 decimals
or p itself. Output: paper/number_trace_report.md listing unmatched tokens with file:line context."""
import glob, json, math, os, re, sys
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(\\?%)?")


def file_values(f):
    """Numeric values contained in one source file (csv/json/md/tex/py)."""
    vals = set()
    def add(x):
        try:
            v = float(str(x).replace(",", ""))
            if math.isfinite(v): vals.add(abs(v))
        except Exception: pass
    if f.endswith(".json"):
        def walk(o):
            if isinstance(o, dict): [walk(v) for v in o.values()]; [add(k) for k in o.keys()]
            elif isinstance(o, list): [walk(v) for v in o]
            else: add(o)
        walk(json.load(open(f)))
    else:
        txt = open(f, errors="ignore").read()
        for m in re.findall(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.?\d*(?:e-?\d+)?", txt):
            add(m.rstrip("."))
    return vals


def pool_values():
    vals = set()
    def add(x):
        try:
            f = float(x)
            if math.isfinite(f): vals.add(abs(f))
        except Exception: pass
    for f in glob.glob(f"{ROOT}/results/**/*.csv", recursive=True) + [f"{ROOT}/paper/figures/gallery_cases.csv"]:
        if "/probes/" in f: continue
        try: d = pd.read_csv(f, header=None, dtype=str)
        except Exception: continue
        for v in d.to_numpy().ravel():
            if isinstance(v, str):
                for m in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", v): add(m)
    def walk(o):
        if isinstance(o, dict): [walk(v) for v in o.values()]; [add(k) for k in o.keys()]
        elif isinstance(o, list): [walk(v) for v in o]
        else: add(o)
    for f in glob.glob(f"{ROOT}/results/**/*.json", recursive=True):
        walk(json.load(open(f)))
    for f in glob.glob(f"{ROOT}/results/RESULTS_*.md") + glob.glob(f"{ROOT}/analysis/PREREGISTRATION*.md") + \
             glob.glob(f"{ROOT}/paper/tables/*.tex") + [f"{ROOT}/geomcheck/probes.py", f"{ROOT}/scripts/gso_inject.py"]:
        for m in re.findall(r"\d[\d,]*\.?\d*(?:e-?\d+)?", open(f).read()):
            add(m.replace(",", ""))
    return vals


def matches(tok_int, dec, pct, pool):
    s = tok_int.replace(",", "") + ("." + dec if dec else "")
    x = float(s); d = len(dec) if dec else 0
    cands = [(x, d)]
    if pct: cands.append((x / 100, d + 2))
    for v in pool:
        for c, dd in cands:
            if abs(round(v, dd) - c) < 10 ** (-dd) / 2 + 1e-12: return True
            if dd == 0 and abs(v - c) < 1e-9: return True
    return False


def strict_check(files):
    """Each block of text ending in a '% src: a; b' comment must match numbers in exactly those files
    (plus method constants in geomcheck/probes.py and the pre-registration files)."""
    base = set()
    for f in [f"{ROOT}/geomcheck/probes.py", f"{ROOT}/scripts/gso_inject.py"] + glob.glob(f"{ROOT}/analysis/PREREGISTRATION*.md"):
        base |= file_values(f)
    out, nocite, n = [], [], 0
    for f in files:
        if not os.path.exists(f): continue
        block = []
        for ln, line in enumerate(open(f), 1):
            m = re.match(r"\s*%\s*src:\s*(.*)", line)
            if m:
                pool = set(base)
                for src in [x.strip() for x in m.group(1).split(";") if x.strip()]:
                    path = f"{ROOT}/{src}" if not src.startswith(("figures/", "tables/")) else f"{ROOT}/paper/{src}"
                    if os.path.exists(path): pool |= file_values(path)
                    else: out.append((os.path.relpath(f, ROOT), ln, "MISSING-SRC", src))
                for (bln, tok, ti, dec, pct, ctx) in block:
                    n += 1
                    if not matches(ti, dec, pct, pool): out.append((os.path.relpath(f, ROOT), bln, tok, ctx))
                block = []
                continue
            for item in tokens(line):
                block.append((ln,) + item)
        for (bln, tok, ti, dec, pct, ctx) in block:
            nocite.append((os.path.relpath(f, ROOT), bln, tok, ctx))
    return out, nocite, n


def tokens(line):
    code = re.sub(r"(?<!\\)%.*", "", line).replace("{,}", ",")
    code = re.sub(r"(Gemini|GPT-|Claude [A-Za-z]+|Mistral Small|Qwen2\.5-VL|Qwen3\.5) ?\d+(\.\d+)?", " ", code)
    code = re.sub(r"\b[0-9a-f]{7}\b", " ", code); code = re.sub(r"\b\d{1,2}:\d{2}\b", " ", code)
    code = re.sub(r"p\{[^}]*\}", " ", code); code = code.replace("One-2-3-45++", " ")
    code = re.sub(r"\\(cite[tp]?|ref|label|input|includegraphics|eqref|url|href|autoref|begin|end)(\[[^\]]*\])?\{[^}]*\}", " ", code)
    code = re.sub(r"\[[^\]]*(width|height|scale)[^\]]*\]", " ", code)
    code = re.sub(r"\\(vspace|hspace|setlength|resizebox|rule)\{[^}]*\}", " ", code)
    res = []
    for m in NUM.finditer(code):
        ti, dec, pct = m.group(1), m.group(2), m.group(3)
        val = float(ti.replace(",", "") + ("." + dec if dec else ""))
        if not dec and not pct and val < 10: continue
        if not dec and not pct and 1990 <= val <= 2030: continue
        res.append((m.group(0), ti, dec, pct, line.strip()[:150]))
    return res


def main(texdir=f"{ROOT}/paper"):
    pool = pool_values()
    files = [f"{texdir}/main.tex"] + sorted(glob.glob(f"{texdir}/sections/*.tex"))
    bad, n = [], 0
    for f in files:
        if not os.path.exists(f): continue
        for ln, line in enumerate(open(f), 1):
            code = re.sub(r"(?<!\\)%.*", "", line).replace("{,}", ",")
            code = re.sub(r"\\(cite[tp]?|ref|label|input|includegraphics|eqref|url|href|autoref|begin|end)(\[[^\]]*\])?\{[^}]*\}", " ", code)
            code = re.sub(r"\[[^\]]*(width|height|scale)[^\]]*\]", " ", code)
            code = re.sub(r"\\(vspace|hspace|setlength|resizebox|rule)\{[^}]*\}", " ", code)
            for m in NUM.finditer(code):
                ti, dec, pct = m.group(1), m.group(2), m.group(3)
                val = float(ti.replace(",", "") + ("." + dec if dec else ""))
                if not dec and not pct and val < 10: continue          # small integers (counts, section refs)
                if not dec and not pct and 1990 <= val <= 2030: continue  # years
                n += 1
                if not matches(ti, dec, pct, pool):
                    bad.append((os.path.relpath(f, ROOT), ln, m.group(0), line.strip()[:160]))
    with open(f"{ROOT}/paper/number_trace_report.md", "w") as fh:
        fh.write(f"# Number trace report\nChecked {n} numeric tokens; {len(bad)} unmatched.\n\n")
        for b in bad: fh.write(f"- `{b[0]}:{b[1]}` **{b[2]}** — {b[3]}\n")
    sbad, nocite, sn = strict_check(files)
    with open(f"{ROOT}/paper/number_trace_report.md", "a") as fh:
        fh.write(f"\n## Strict mode (numbers must match the files named in the following '% src:' comment)\n"
                 f"Checked {sn} tokens; {len(sbad)} unmatched; {len(nocite)} numeric tokens not followed by a src comment.\n\n")
        for b in sbad: fh.write(f"- `{b[0]}:{b[1]}` **{b[2]}** — {b[3]}\n")
        fh.write("\n### Tokens without a src comment (checked only against the global pool above)\n")
        for b in nocite: fh.write(f"- `{b[0]}:{b[1]}` **{b[2]}** — {b[3]}\n")
    print(f"strict: checked {sn}, unmatched {len(sbad)}, no-src {len(nocite)}")
    for b in sbad: print("STRICT", b)
    print(f"checked {n}, unmatched {len(bad)}")
    for b in bad: print(b)


if __name__ == "__main__":
    main()
