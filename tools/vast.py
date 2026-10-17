"""Deploy and run training on a vast.ai GPU instance.

Instances are named (`--name`, default "main") and remembered in .vast/<name>.json, so several
experiments can run on separate instances; every command targets the named instance.
Requires VAST_API_KEY (loaded from .env by just) and a local SSH key (~/.ssh/id_ed25519.pub).
"""

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / ".vast"
STATE = STATE_DIR / "main.json"  # replaced by main() with the --name instance's file
REMOTE_DIR = "/root/starship"
SSH_KEY = Path.home() / ".ssh" / "id_ed25519"
DEFAULT_IMAGE = "nvidia/cuda:12.8.1-base-ubuntu24.04"
TB_LOCAL_PORT = 16006
# The locked torch wheels are built against CUDA 13, so the host driver must support it.
CUDA_CHECK = (
    "import warp as wp, torch; wp.init(); "
    "print('warp cuda devices:', wp.get_cuda_device_count(), "
    "'| torch cuda:', torch.cuda.is_available())"
)
BASE_QUERY = (
    "num_gpus=1 reliability>0.99 cuda_vers>=13.0 inet_down>=500 rentable=true verified=true"
)
RSYNC_EXCLUDES = [
    ".venv",
    ".git",
    ".direnv",
    "logs",
    "out",
    ".vast",
    ".env",
    "__pycache__",
    "*.rrd",
]


def vast(*args: str):
    key = os.environ.get("VAST_API_KEY")
    if not key:
        sys.exit("VAST_API_KEY is not set (put it in .env, see .env.example)")
    cmd = ["vastai", "--api-key", key, *args, "--raw"]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return json.loads(out) if out.strip() else None


def load_state() -> dict:
    if not STATE.exists():
        sys.exit(f"no instance named {STATE.stem!r}; run `just vast up --name {STATE.stem}` first")
    return json.loads(STATE.read_text())


def tb_port(state: dict) -> int:
    return state.get("tb_port", TB_LOCAL_PORT)


def free_tb_port() -> int:
    """Local tunnel port not used by another named instance."""
    used = {tb_port(json.loads(p.read_text())) for p in STATE_DIR.glob("*.json")}
    port = TB_LOCAL_PORT
    while port in used:
        port += 1
    return port


def save_state(state: dict) -> None:
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2))


def ssh_base(state: dict) -> list[str]:
    return [
        "ssh",
        "-i",
        str(SSH_KEY),
        "-p",
        str(state["ssh_port"]),
        # Only the key file: an SSH agent (e.g. Secretive) would prompt for every connection.
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "IdentityAgent=none",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ServerAliveInterval=30",
        f"root@{state['ssh_host']}",
    ]


def remote(state: dict, command: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([*ssh_base(state), command], check=check)


def search(gpu: str, limit: int, max_price: float | None) -> list[dict]:
    query = f"{BASE_QUERY} gpu_name={gpu}"
    if max_price is not None:
        query += f" dph_total<={max_price}"
    return vast("search", "offers", query, "-o", "dph_total", "--limit", str(limit))


def print_offers(offers: list[dict]) -> None:
    print(f"{'offer':>10}  {'gpu':<10} {'$/h':>6}  {'cuda':>5}  {'rel':>5}  {'cpus':>4}  location")
    for o in offers:
        print(
            f"{o['id']:>10}  {o['gpu_name']:<10} {o['dph_total']:>6.3f}  {o['cuda_max_good']:>5}"
            f"  {o['reliability2']:>5.3f}  {o['cpu_cores_effective']:>4.0f}  {o.get('geolocation')}"
        )


def ensure_ssh_key() -> None:
    pub = SSH_KEY.with_suffix(".pub").read_text().strip()
    registered = [k["public_key"].strip() for k in vast("show", "ssh-keys") or []]
    if not any(k.split()[:2] == pub.split()[:2] for k in registered):
        print(f"registering {SSH_KEY}.pub with vast.ai")
        vast("create", "ssh-key", pub)


def ssh_routes(info: dict) -> list[tuple[str, int]]:
    """Direct connection (public IP and mapped port 22) first, then vast.ai's SSH proxy, which
    sometimes stops answering while the instance keeps running."""
    routes = []
    mapped = (info.get("ports") or {}).get("22/tcp") or []
    if info.get("public_ipaddr") and mapped:
        routes.append((info["public_ipaddr"].strip(), int(mapped[0]["HostPort"])))
    if info.get("ssh_host"):
        routes.append((info["ssh_host"], int(info["ssh_port"])))
    return routes


def reachable(state: dict) -> bool:
    *opts, host = ssh_base(state)
    probe = [*opts, "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host, "true"]
    return subprocess.run(probe, capture_output=True).returncode == 0


def wait_for_ssh(instance_id: int, timeout: float = 900.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = vast("show", "instance", str(instance_id))
        status = info.get("actual_status")
        if status == "running":
            for host, port in ssh_routes(info):
                state = {
                    "id": instance_id,
                    "ssh_host": host,
                    "ssh_port": port,
                    "gpu": info.get("gpu_name"),
                    "dph": info.get("dph_total"),
                }
                if reachable(state):
                    return state
        print(f"  waiting for instance {instance_id} (status: {status})")
        time.sleep(15)
    sys.exit(f"instance {instance_id} not reachable after {timeout:.0f}s; check `just vast list`")


def cmd_offers(args) -> None:
    print_offers(search(args.gpu, args.limit, args.max_price))


def cmd_up(args) -> None:
    if STATE.exists():
        state = load_state()
        if "ssh_host" in state:
            sys.exit(f"an instance is already active ({state['id']}); `just vast down` first")
        print(f"resuming setup of instance {state['id']}")
        _finish_up(args, state["id"])
        return
    ensure_ssh_key()
    if args.offer is None:
        offers = search(args.gpu, 1, args.max_price)
        if not offers:
            sys.exit(f"no offers for {args.gpu}")
        print_offers(offers)
        args.offer = offers[0]["id"]
    print(f"creating instance from offer {args.offer} ({args.image}, {args.disk} GB disk)")
    created = vast(
        "create", "instance", str(args.offer),
        "--image", args.image, "--disk", str(args.disk),
        "--ssh", "--direct", "--label", "starship",
    )  # fmt: skip
    instance_id = created["new_contract"]
    save_state({"id": instance_id, "tb_port": free_tb_port()})
    _finish_up(args, instance_id)


def _finish_up(args, instance_id: int) -> None:
    previous = json.loads(STATE.read_text()) if STATE.exists() else {}
    state = {**previous, **wait_for_ssh(instance_id)}
    save_state(state)
    print(f"instance {instance_id} is up: {state['gpu']} at ${state['dph']:.3f}/h")
    cmd_push(args)
    cmd_setup(args)


def cmd_push(args) -> None:
    state = load_state()
    excludes = [f"--exclude={e}" for e in RSYNC_EXCLUDES]
    ssh = " ".join(shlex.quote(p) for p in ssh_base(state)[:-1])
    target = f"root@{state['ssh_host']}:{REMOTE_DIR}/"
    subprocess.run(
        ["rsync", "-az", "--delete", *excludes, "-e", ssh, f"{ROOT}/", target], check=True
    )
    print(f"pushed code to {target}")


def cmd_setup(args, timeout: float = 1800.0) -> None:
    state = load_state()
    tools = """
set -e
command -v rsync >/dev/null && command -v tmux >/dev/null && command -v curl >/dev/null || \
  (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync tmux curl)
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
"""
    remote(state, tools)
    # The dependency install takes minutes; run it in tmux on the instance so a dropped SSH
    # connection cannot kill it, and poll for its exit code.
    job = (
        f"export PATH=$HOME/.local/bin:$PATH && cd {REMOTE_DIR} && "
        f'(uv sync --locked -q && uv run python -c "{CUDA_CHECK}") > /root/setup.log 2>&1; '
        "echo $? > /root/setup.done"
    )
    remote(
        state,
        "rm -f /root/setup.done; tmux kill-session -t setup 2>/dev/null; "
        f"tmux new-session -d -s setup {shlex.quote(job)}",
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(15)
        probe = subprocess.run(
            [*ssh_base(state), "cat /root/setup.done 2>/dev/null"], capture_output=True, text=True
        )
        code = probe.stdout.strip()
        if code:
            remote(state, "tail -n 3 /root/setup.log", check=False)
            if code != "0":
                sys.exit(f"setup failed with exit code {code}; see /root/setup.log on the instance")
            return
        print("  installing dependencies...")
    sys.exit("setup did not finish in time; check /root/setup.log on the instance")


def cmd_train(args) -> None:
    state = load_state()
    cmd_push(args)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    overrides = " ".join(shlex.quote(a) for a in args.overrides)
    job = (
        f"cd {REMOTE_DIR} && mkdir -p logs && export PATH=$HOME/.local/bin:$PATH && "
        f"uv run starship-train {overrides} 2>&1 | tee logs/train_{stamp}.log"
    )
    exists = remote(state, "tmux has-session -t train 2>/dev/null", check=False).returncode == 0
    if exists:
        sys.exit("a training session is already running; `just vast attach` or `just vast stop`")
    remote(state, f"tmux new-session -d -s train {shlex.quote(job)}")
    print(f"training started in tmux session 'train' (logs/train_{stamp}.log)")


def cmd_exec(args) -> None:
    state = load_state()
    cmd_push(args)
    command = " ".join(shlex.quote(a) for a in args.command)
    result = remote(
        state, f"cd {REMOTE_DIR} && export PATH=$HOME/.local/bin:$PATH && {command}", check=False
    )
    sys.exit(result.returncode)


def cmd_bench(args) -> None:
    args.command = ["uv", "run", "starship-bench", *args.bench_args]
    cmd_exec(args)


def cmd_status(args) -> None:
    state = load_state()
    info = vast("show", "instance", str(state["id"]))
    print(
        f"instance {state['id']}: {info.get('actual_status')} | {info.get('gpu_name')} | "
        f"${info.get('dph_total', 0):.3f}/h | ssh -p {state['ssh_port']} root@{state['ssh_host']}"
    )
    remote(
        state,
        "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu "
        "--format=csv,noheader; echo 'tmux:'; tmux ls 2>/dev/null || echo '  none'; "
        f"latest=$(ls -t {REMOTE_DIR}/logs/train_*.log 2>/dev/null | head -1); "
        '[ -n "$latest" ] && echo "--- $latest" && tail -n 5 "$latest"',
        check=False,
    )


def port_open(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def cmd_logs(args) -> None:
    state = load_state()
    tb = shlex.quote(
        f"cd {REMOTE_DIR} && mkdir -p logs && "
        "$HOME/.local/bin/uv run tensorboard --logdir logs --port 6006"
    )
    remote(state, f"tmux has-session -t tb 2>/dev/null || tmux new-session -d -s tb {tb}")
    port = tb_port(state)
    if not port_open(port):
        subprocess.run(
            [*ssh_base(state)[:-1], "-f", "-N", "-L", f"{port}:localhost:6006",
             f"root@{state['ssh_host']}"],
            check=True,
        )  # fmt: skip
    url = f"http://localhost:{port}"
    for _ in range(60):
        try:
            urllib.request.urlopen(url, timeout=2)
            break
        except OSError:
            time.sleep(1)
    print(f"remote TensorBoard at {url}")
    webbrowser.open(url)


def cmd_pull(args) -> None:
    state = load_state()
    ssh = " ".join(shlex.quote(p) for p in ssh_base(state)[:-1])
    source = f"root@{state['ssh_host']}:{REMOTE_DIR}/logs/"
    (ROOT / "logs").mkdir(exist_ok=True)
    subprocess.run(["rsync", "-az", "-e", ssh, source, f"{ROOT}/logs/"], check=True)
    print(f"pulled remote logs into {ROOT / 'logs'}")


def fetch_recordings(state: dict, count: int) -> list[Path]:
    """Copy the newest finished eval recordings of any run into local logs/; returns new files."""
    # Skip files touched in the last minute: the recorder may still be writing them.
    listing = subprocess.run(
        [
            *ssh_base(state),
            f"cd {REMOTE_DIR}/logs && find . -path '*/rerun/*.rrd' -mmin +1 -printf '%T@ %P\\n' "
            f"| sort -rn | head -{count} | cut -d' ' -f2-",
        ],
        capture_output=True,
        text=True,
    )
    fresh = [p for p in listing.stdout.split() if not (ROOT / "logs" / p).exists()]
    if not fresh:
        return []
    ssh = " ".join(shlex.quote(p) for p in ssh_base(state)[:-1])
    source = f"root@{state['ssh_host']}:{REMOTE_DIR}/logs/./"
    files = [source + p for p in fresh]
    subprocess.run(["rsync", "-az", "--relative", "-e", ssh, *files, f"{ROOT}/logs/"], check=True)
    return [ROOT / "logs" / p for p in fresh]


def cmd_rerun(args) -> None:
    """Open the newest eval recordings in the Rerun viewer. A running viewer receives them as
    extra recordings, so watching keeps one window."""
    state = load_state()
    while True:
        try:
            files = fetch_recordings(state, args.count)
        except subprocess.CalledProcessError as e:
            print(f"could not fetch recordings: {e}")
            files = []
        for f in sorted(files):
            print(f"opening {f.relative_to(ROOT)}")
            subprocess.Popen(
                ["rerun", str(f)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        if not args.watch:
            if not files:
                print("no new recordings")
            return
        time.sleep(args.watch * 60)


def cmd_reconnect(args) -> None:
    """Re-resolve how to reach the instance, e.g. when the SSH proxy stops answering."""
    state = load_state()
    state.update(wait_for_ssh(state["id"], timeout=120))
    save_state(state)
    print(f"reachable via {state['ssh_host']}:{state['ssh_port']}")


def cmd_ssh(args) -> None:
    os.execvp("ssh", ssh_base(load_state()))


def cmd_attach(args) -> None:
    *opts, host = ssh_base(load_state())
    os.execvp("ssh", [*opts, "-t", host, "tmux attach -t train"])


def cmd_stop(args) -> None:
    remote(load_state(), "tmux kill-session -t train", check=False)


def cmd_list(args) -> None:
    names = {json.loads(p.read_text()).get("id"): p.stem for p in STATE_DIR.glob("*.json")}
    for i in vast("show", "instances") or []:
        print(
            f"{i['id']:>10}  {i.get('actual_status')!s:<10} {i.get('gpu_name')!s:<10} "
            f"${i.get('dph_total', 0):.3f}/h  {names.get(i['id'], '(not tracked)')}"
        )


def cmd_down(args) -> None:
    state = load_state()
    subprocess.run(["pkill", "-f", f"{tb_port(state)}:localhost:6006"], check=False)
    # --yes: the CLI otherwise waits for an interactive confirmation it never gets.
    vast("destroy", "instance", str(state["id"]), "--yes")
    STATE.unlink()
    print(f"destroyed instance {state['id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="main", help="instance name (default: main)")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    def add(name, fn, help):
        p = sub.add_parser(name, help=help)
        p.set_defaults(fn=fn)
        return p

    for name, fn, help in [
        ("offers", cmd_offers, "list the cheapest matching offers"),
        ("up", cmd_up, "rent the cheapest matching offer, push code and set it up"),
    ]:
        p = add(name, fn, help)
        p.add_argument("--gpu", default="RTX_3090", help="vast gpu_name, spaces as _")
        p.add_argument("--max-price", type=float, default=None, help="max $/h")
        if name == "offers":
            p.add_argument("--limit", type=int, default=10)
        else:
            p.add_argument("--offer", type=int, default=None, help="rent this offer id instead")
            p.add_argument("--image", default=DEFAULT_IMAGE)
            p.add_argument("--disk", type=int, default=40, help="GB")
    add("push", cmd_push, "rsync the working tree to the instance")
    add("setup", cmd_setup, "install uv and dependencies on the instance")
    p = add("train", cmd_train, "push and start starship-train in tmux session 'train'")
    p.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. num_envs=16384")
    p = add("exec", cmd_exec, "push and run a command in the remote repo, streaming output")
    p.add_argument("command_args", nargs="*")
    p = add("bench", cmd_bench, "push and run starship-bench (find the best num_envs)")
    p.add_argument("bench_args", nargs="*", help="e.g. --envs 8192,32768 model=mlp")
    add("status", cmd_status, "instance, GPU utilization, sessions and latest training log")
    add("logs", cmd_logs, f"remote TensorBoard via SSH tunnel on localhost:{TB_LOCAL_PORT}")
    add("pull", cmd_pull, "rsync remote logs/ (events, policies) into local logs/")
    p = add("rerun", cmd_rerun, "pull the newest eval recordings and open them in Rerun")
    p.add_argument("--count", type=int, default=2, help="newest recordings to open")
    p.add_argument("--watch", type=float, default=0, help="keep checking every N minutes")
    add("ssh", cmd_ssh, "interactive shell on the instance")
    add("attach", cmd_attach, "attach to the training tmux session")
    add("stop", cmd_stop, "stop the training session")
    add("reconnect", cmd_reconnect, "re-resolve the SSH route (direct first, then proxy)")
    add("list", cmd_list, "all instances on the account")
    add("down", cmd_down, "destroy the active instance")

    sys.stdout.reconfigure(line_buffering=True)
    # Unknown options belong to the remote command for exec and bench (e.g. --envs).
    args, extra = parser.parse_known_args()
    global STATE
    STATE = STATE_DIR / f"{args.name}.json"
    legacy = STATE_DIR / "instance.json"
    if legacy.exists() and not (STATE_DIR / "main.json").exists():
        legacy.rename(STATE_DIR / "main.json")
    if args.fn is cmd_exec:
        args.command = [*args.command_args, *extra]
    if extra and args.fn is cmd_bench:
        args.bench_args = [*extra, *args.bench_args]
    elif extra and args.fn is not cmd_exec:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    args.fn(args)


if __name__ == "__main__":
    main()
