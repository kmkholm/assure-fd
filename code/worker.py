"""One grid worker. Several workers share the grid through per-run lock files (run_grid.py); each worker retries
after a crash and resumes, since finished runs are skipped. Usage: python worker.py <id> [reverse]"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "..", "launch_full.log")
wid = sys.argv[1] if len(sys.argv) > 1 else "1"
reverse = len(sys.argv) > 2 and sys.argv[2] == "reverse"
sys.path.insert(0, HERE)
os.environ.setdefault("ASSURE_FULL", "1")
import data as D  # noqa: E402

order = list(reversed(D.SCENARIOS)) if reverse else list(D.SCENARIOS)


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + f"[w{wid}] " + msg + "\n")


log("worker started, pid %d, order %s" % (os.getpid(), "reverse" if reverse else "forward"))
for attempt in range(6):
    p = subprocess.Popen([sys.executable, "-u", "run_grid.py", "--scenario", *order, "--seeds", "5", "--epochs", "40"], cwd=HERE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    for line in p.stdout:
        line = line.rstrip()
        if line.startswith(("done", "Traceback")) or "Error" in line:
            log(line)
    p.wait()
    log("run_grid exit %d" % p.returncode)
    if p.returncode == 0:
        break
    time.sleep(30)
log("worker finished")
