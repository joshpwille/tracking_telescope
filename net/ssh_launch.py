# net/ssh_launch.py
from __future__ import annotations
import subprocess
import shlex
import time
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict

# ------------------ Node spec ------------------

@dataclass
class Node:
    host: str                       # e.g., "192.168.1.121" or "jetson.local"
    user: str = "ubuntu"
    port: int = 22
    identity: Optional[str] = None  # path to SSH key, e.g., "~/.ssh/id_rsa"
    python: str = "python3"
    venv: str = ""                  # e.g., "/home/ubuntu/venv/bin/activate"
    workdir: str = ""               # e.g., "/home/ubuntu/project"
    script: str = "/home/ubuntu/project/app/remote_capture.py"
    args: Union[str, List[str]] = ""  # extra CLI args
    env: Dict[str, str] = field(default_factory=dict)  # extra env vars for remote
    use_tmux: bool = True
    tmux_session: str = "capture"   # tmux session name
    log_path: Optional[str] = None  # e.g., "/home/ubuntu/project/outputs/capture.log"

# ------------------ Helpers ------------------

def _ssh_base_cmd(node: Node) -> List[str]:
    cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-p", str(node.port),
    ]
    if node.identity:
        cmd += ["-i", node.identity]
    cmd += [f"{node.user}@{node.host}"]
    return cmd

def _wrap_remote_command(node: Node, remote_cmd: str) -> str:
    """
    Build the final remote command string:
    - cd workdir (optional)
    - source venv (optional)
    - export ENV (optional)
    - run 'remote_cmd'
    """
    parts = []
    if node.workdir:
        parts.append(f"cd {shlex.quote(node.workdir)}")
    if node.venv:
        parts.append(f"source {shlex.quote(node.venv)}")
    for k, v in node.env.items():
        parts.append(f"export {k}={shlex.quote(v)}")
    parts.append(remote_cmd)
    return " && ".join(parts)

def _tmux_wrap(node: Node, label: str, cmd: str) -> str:
    """
    Run the command in a tmux session so it survives SSH disconnects.
    Creates the session if missing; then sends the command to a new window.
    """
    sess = shlex.quote(node.tmux_session)
    window_name = shlex.quote(label)
    log_redir = ""
    if node.log_path:
        log_redir = f" |& tee -a {shlex.quote(node.log_path)}"
    return (
        # create session if it doesn't exist
        f"(tmux has-session -t {sess} 2>/dev/null || tmux new-session -d -s {sess}) && "
        # create a new window with a descriptive name and run the command
        f"tmux new-window -t {sess} -n {window_name} '{cmd}{log_redir}'"
    )

def ping_ms(host: str, count: int = 3, timeout: int = 2) -> float:
    try:
        out = subprocess.run(
            ["ping", "-c", str(count), "-W", str(timeout), host],
            capture_output=True, text=True, check=False
        )
        for line in out.stdout.splitlines():
            if "min/avg/max" in line:
                # works for both Linux/BSD variants
                stats = line.split("=")[1].strip().split()[0].split("/")
                return float(stats[1])
    except Exception:
        pass
    return -1.0

def ssh_run(node: Node, remote_cmd: str):
    base = _ssh_base_cmd(node)
    return subprocess.Popen(base + [remote_cmd])

def ssh_exec(node: Node, remote_cmd: str, timeout: float = 5.0) -> subprocess.CompletedProcess:
    base = _ssh_base_cmd(node)
    return subprocess.run(base + [remote_cmd], capture_output=True, text=True, timeout=timeout)

def get_remote_time_ns(node: Node) -> Optional[int]:
    """
    Ask the remote for its time in ns via python (more portable than `date +%s%N`).
    Returns None if it fails.
    """
    try:
        rc = ssh_exec(node, f"{node.python} - <<'PY'\nimport time; print(time.time_ns())\nPY", timeout=5.0)
        if rc.returncode == 0:
            return int(rc.stdout.strip())
    except Exception:
        pass
    return None

def _args_to_str(args: Union[str, List[str]]) -> str:
    if isinstance(args, str):
        return args
    return " ".join(shlex.quote(a) for a in args)

# ------------------ Launcher ------------------

def launch(nodes: List[Node], warmup_s: float = 2.0, start_delay_s: float = 1.0, label: str = "run"):
    """
    Launch the capture script on each node with a per-node absolute start time.

    Steps:
      1) Ping each node (informational).
      2) Read each node's remote clock (time_ns).
      3) For each node, compute start_at_ns_remote = remote_now_ns + (warmup_s + start_delay_s)*1e9
      4) Build the command (venv/workdir/env respected), optionally via tmux, and run.

    Returns: list of Popen handles (if not using tmux), otherwise [].
    """
    procs = []

    print("[ssh] Measuring connectivity and clocks...")
    for n in nodes:
        rtt = ping_ms(n.host)
        print(f"[ssh] {n.host} RTT ~ {rtt:.1f} ms")

    # Determine a baseline controller time (info only)
    controller_now = time.time_ns()

    for n in nodes:
        remote_now = get_remote_time_ns(n)
        if remote_now is None:
            print(f"[ssh] WARNING: failed to read remote clock on {n.host}; falling back to controller clock.")
            # Fall back: assume minimal skew; remote will use its own now() at command time
            remote_now = controller_now

        # Per-node absolute start time on the REMOTE clock
        start_at_ns = remote_now + int((warmup_s + start_delay_s) * 1e9)

        # Build the remote python command
        args_str = _args_to_str(n.args)
        base_cmd = f"{shlex.quote(n.python)} {shlex.quote(n.script)} --start-at-ns {start_at_ns}"
        if args_str:
            base_cmd += f" {args_str}"

        # Allow passing START_AT_NS in env as well
        env_cmd = f"export START_AT_NS={start_at_ns}"

        remote_cmd = _wrap_remote_command(n, f"{env_cmd} && {base_cmd}")

        if n.use_tmux:
            final_cmd = _tmux_wrap(n, label=label, cmd=remote_cmd)
            print(f"[ssh] {n.host} (tmux:{n.tmux_session}) :: {base_cmd}")
            procs.append(ssh_run(n, final_cmd))
        else:
            # Non-tmux: keep the SSH session attached; recommend log_path for output
            if n.log_path:
                remote_cmd += f" |& tee -a {shlex.quote(n.log_path)}"
            print(f"[ssh] {n.host} :: {base_cmd}")
            procs.append(ssh_run(n, remote_cmd))

    return procs

# ------------------ Utilities ------------------

def kill_tmux(nodes: List[Node]) -> None:
    """Kill tmux sessions specified on nodes (only if use_tmux=True)."""
    for n in nodes:
        if not n.use_tmux:
            continue
        try:
            ssh_exec(n, f"tmux kill-session -t {shlex.quote(n.tmux_session)}", timeout=5.0)
            print(f"[ssh] {n.host} :: tmux session '{n.tmux_session}' killed.")
        except Exception as e:
            print(f"[ssh] {n.host} :: tmux kill failed: {e}")

def list_tmux(nodes: List[Node]) -> None:
    """List tmux sessions on nodes."""
    for n in nodes:
        try:
            rc = ssh_exec(n, "tmux ls || true", timeout=5.0)
            print(f"\n[ssh] {n.host} tmux:\n{rc.stdout.strip()}")
        except Exception as e:
            print(f"[ssh] {n.host} :: tmux ls failed: {e}")

# ------------------ Example (optional) ------------------

if __name__ == "__main__":
    # Example usage: customize to your hosts and paths.
    nodes = [
        Node(
            host="192.168.1.121",
            user="ubuntu",
            identity="~/.ssh/id_ed25519",
            workdir="/home/ubuntu/project",
            venv="/home/ubuntu/venv/bin/activate",
            script="/home/ubuntu/project/app/remote_capture.py",
            args=["--src", "0", "--w", "1920", "--h", "1080", "--fps", "30"],
            log_path="/home/ubuntu/project/outputs/left.log",
            tmux_session="stereo",
        ),
        Node(
            host="192.168.1.122",
            user="ubuntu",
            identity="~/.ssh/id_ed25519",
            workdir="/home/ubuntu/project",
            venv="/home/ubuntu/venv/bin/activate",
            script="/home/ubuntu/project/app/remote_capture.py",
            args=["--src", "0", "--w", "1920", "--h", "1080", "--fps", "30"],
            log_path="/home/ubuntu/project/outputs/right.log",
            tmux_session="stereo",
        ),
    ]

    # Launch both with a per-node absolute start time (remote clocks)
    launch(nodes, warmup_s=2.0, start_delay_s=1.0, label="capture")

    # Helpful utilities:
    # list_tmux(nodes)
    # kill_tmux(nodes)
