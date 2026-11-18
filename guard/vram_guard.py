"""VRAM monitor for the Arc Pro B50 (xe, PCI 0000:03:00.0).

GET /vram -> {used_gib, total_gib, free_gib, procs: [{pid, comm, mib}], age_s}

Per-process VRAM comes from /proc/<pid>/fdinfo (drm-total-vram0). torch.xpu.mem_get_info() is not
supported on this card, so this is the only host-wide view. Runs with the host PID namespace and
CAP_SYS_PTRACE (needed to read root-owned processes' fdinfo); /proc is mounted read-only at /hp.
Driver-internal allocations are not counted, so free_gib is an upper bound.
"""
import collections, glob, json, os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PDEV = os.environ.get("PDEV", "0000:03:00.0")
TOTAL_GIB = float(os.environ.get("TOTAL_GIB", "16"))
PROC = os.environ.get("PROC", "/hp")
_state = {"t": 0, "data": None}
_lock = threading.Lock()


def scan():
    tot, seen = collections.Counter(), set()
    for fdi in glob.glob(f"{PROC}/[0-9]*/fdinfo/*"):
        pid = fdi.split("/")[-3]
        try:
            txt = open(fdi).read()
        except Exception:
            continue
        if f"drm-pdev:\t{PDEV}" not in txt:
            continue
        kv = dict(l.split(":", 1) for l in txt.splitlines() if ":" in l)
        if "drm-total-vram0" not in kv or (pid, kv.get("drm-client-id")) in seen:
            continue
        seen.add((pid, kv.get("drm-client-id")))
        tot[pid] += int(kv["drm-total-vram0"].split()[0])
    procs = []
    for pid, kib in tot.most_common():
        try:
            comm = open(f"{PROC}/{pid}/comm").read().strip()
        except Exception:
            comm = "?"
        procs.append({"pid": int(pid), "comm": comm, "mib": round(kib / 1024)})
    used = sum(tot.values()) / 1024 / 1024
    return {"used_gib": round(used, 2), "total_gib": TOTAL_GIB, "free_gib": round(TOTAL_GIB - used, 2), "procs": procs}


def get(max_age=2.0):
    with _lock:
        if time.time() - _state["t"] > max_age:
            _state["data"], _state["t"] = scan(), time.time()
        return {**_state["data"], "age_s": round(time.time() - _state["t"], 1)}


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        b = json.dumps(get()).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8792))
    print(f"vram guard on 127.0.0.1:{port}: {json.dumps(get())}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
