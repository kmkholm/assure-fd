"""Print the per-scenario values written to paper/tables/macros.tex (for writing and checking the text)."""
import re

d = {}
for line in open("../paper/tables/macros.tex", encoding="utf-8"):
    line = line.strip()
    m = re.match(r"\\expandafter\\def\\csname r@(.+?)@(.+?)@(.+?)\\endcsname\{(.*)\}", line)
    if m:
        d[(m.group(1), m.group(2), m.group(3))] = m.group(4)
        continue
    m2 = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}", line)
    if m2:
        print(m2.group(1), "=", m2.group(2))
scen = sorted({k[0] for k in d if not k[0].startswith("cmapss")})
for s in scen:
    g = lambda v, m: d.get((s, v, m), "-")
    print(f"{s:16s} acc {g('no_tta', 'acc')}->{g('full', 'acc')} | AUC {g('auc', 'src')}->{g('auc', 'tta')} | full far {g('full', 'far')} "
          f"miss {g('full', 'miss')} insp {g('full', 'insp')} cost {g('full', 'cost')} | k20 miss {g('k20', 'miss')} insp {g('k20', 'insp')} "
          f"cost {g('k20', 'cost')} | dann far {g('dann_softmax', 'far')} cost {g('dann_softmax', 'cost')} | dann+cp far {g('dann_conformal', 'far')} "
          f"miss {g('dann_conformal', 'miss')} | covh {g('full', 'covh')} covf {g('full', 'covf')} k50 {g('k50', 'covf')}")
for k, v in d.items():
    if k[0].startswith("cmapss"):
        print(k, v)
