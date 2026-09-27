"""Fetch title/authors/year for arXiv ids from the arXiv API (verification aid for refs.bib)."""
import sys, json, urllib.request, xml.etree.ElementTree as ET, time
ids = sys.argv[1].split(",")
ns = {"a": "http://www.w3.org/2005/Atom"}
out = {}
for i in range(0, len(ids), 20):
    u = "https://export.arxiv.org/api/query?max_results=50&id_list=" + ",".join(ids[i:i+20])
    root = ET.fromstring(urllib.request.urlopen(u, timeout=60).read())
    for e in root.findall("a:entry", ns):
        aid = e.find("a:id", ns).text.split("/abs/")[-1].split("v")[0]
        out[aid] = dict(title=" ".join(e.find("a:title", ns).text.split()),
                        authors=[a.find("a:name", ns).text for a in e.findall("a:author", ns)],
                        year=e.find("a:published", ns).text[:4])
    time.sleep(3)
json.dump(out, open("paper/arxiv_meta.json", "w"), indent=1)
for k, v in out.items(): print(k, v["year"], v["title"], "|", ", ".join(v["authors"][:12]), "..." if len(v["authors"]) > 12 else "")
