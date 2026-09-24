#!/usr/bin/env python3
"""Run isolated storage faults; missing prerequisites and missed faults fail."""

from __future__ import annotations

import argparse
import contextlib
import errno
import json
import os
import random
import shutil
import signal
import struct
import subprocess
import sys
import time
import traceback
import uuid
import xml.etree.ElementTree as ET
from functools import partial
from pathlib import Path


class Suite:
    def __init__(self, output):
        self.output = output
        self.results = []

    def case(self, name, action):
        folder = self.output / name
        folder.mkdir()
        start = time.monotonic()
        error = None
        try:
            action(folder)
        except Exception:
            error = traceback.format_exc()
            (folder / "failure.txt").write_text(error)
            sys.stderr.write(error + "\n")
            sys.stderr.flush()
        self.results.append((name, time.monotonic() - start, error))
        sys.stdout.write(f"{'FAIL' if error else 'PASS'} {name}\n")
        sys.stdout.flush()
        self.write_report()

    def write_report(self):
        root = ET.Element(
            "testsuite",
            name="storage-stability",
            tests=str(len(self.results)),
            failures=str(sum(bool(r[2]) for r in self.results)),
        )
        for name, seconds, error in self.results:
            case = ET.SubElement(root, "testcase", name=name, time=f"{seconds:.3f}")
            if error:
                ET.SubElement(
                    case, "failure", message=error.splitlines()[-1]
                ).text = error
        ET.ElementTree(root).write(
            self.output / "junit.xml", encoding="utf-8", xml_declaration=True
        )


def command(*args, log=None):
    result = subprocess.run(
        [str(a) for a in args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
    )
    if log:
        with log.open("a") as stream:
            stream.write(f"$ {' '.join(map(str, args))}\n{result.stdout}\n")
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {args}\n{result.stdout}"
        )
    return result.stdout.strip()


class Volume:
    """Own only newly allocated loop devices and a uniquely named dm target."""

    def __init__(self, folder):
        self.folder = folder
        self.mountpoint = folder / "mnt"
        self.image = folder / "data.img"
        self.loops = []
        self.mapper = None
        self.mounted = False
        self.log = folder / "commands.log"

    def root(self, *args):
        return command("sudo", "-n", *args, log=self.log)

    def loop(self, image):
        device = self.root("losetup", "--find", "--show", image)
        self.loops.append(device)
        return device

    def mount(self, device):
        self.root("mount", "-o", "noatime", device, self.mountpoint)
        self.mounted = True
        self.root("chown", f"{os.getuid()}:{os.getgid()}", self.mountpoint)

    def unmount(self):
        if self.mounted:
            self.root("umount", self.mountpoint)
            self.mounted = False

    def __enter__(self):
        self.mountpoint.mkdir()
        with self.image.open("wb") as stream:
            stream.truncate(128 * 1024 * 1024)
        try:
            self.device = self.loop(self.image)
            self.root("mkfs.ext4", "-q", "-F", "-m", "0", "-N", "1024", self.device)
            self.mount(self.device)
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self, exc_type, exc, tb):
        errors = []
        try:
            self.unmount()
        except Exception as error:
            errors.append(str(error))
        if self.mapper:
            try:
                self.root("dmsetup", "remove", self.mapper)
            except Exception as error:
                errors.append(str(error))
        for device in reversed(self.loops):
            try:
                self.root("losetup", "--detach", device)
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError("Storage cleanup failed: " + "\n".join(errors)) from exc
        if exc_type is None:
            # Keep failed images; successful cases only need logs and metadata.
            for image in self.folder.glob("*.img"):
                image.unlink()


def worker_args(args, database, mode, fts, minimum=64):
    return [
        str(args.worker),
        str(database),
        mode,
        "fts" if fts else "plain",
        str(minimum),
    ]


def run_worker(args, database, mode, fts, folder, minimum=64):
    log = folder / f"{mode}.log"
    with log.open("w") as output:
        result = subprocess.run(
            worker_args(args, database, mode, fts, minimum),
            check=False,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=90,
        )
    if result.returncode:
        raise RuntimeError(f"{mode} failed ({result.returncode}):\n{log.read_text()}")


@contextlib.contextmanager
def paused_worker(args, database, operation, fts, folder, env=None):
    log = folder / "operation.log"
    with log.open("w") as output:
        process = subprocess.Popen(
            worker_args(args, database, operation, fts),
            env={**os.environ, **(env or {})},
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                pid, status = os.waitpid(process.pid, os.WNOHANG | os.WUNTRACED)
                if pid:
                    if os.WIFSTOPPED(status) and os.WSTOPSIG(status) == signal.SIGSTOP:
                        if "READY\n" not in log.read_text():
                            raise RuntimeError(
                                "Worker stopped without READY acknowledgement"
                            )
                        break
                    process.returncode = os.waitstatus_to_exitcode(status)
                    raise RuntimeError(
                        f"Worker exited before checkpoint:\n{log.read_text()}"
                    )
                time.sleep(0.01)
            else:
                raise TimeoutError("Worker did not reach checkpoint within 60 seconds")
            yield process
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)


def resume(process):
    os.kill(process.pid, signal.SIGCONT)
    return process.wait(timeout=90)


def fill(volume, kind):
    filler = volume.mountpoint / "filler"
    if kind == "blocks":
        with filler.open("wb", buffering=0) as stream:
            block = bytes(1024 * 1024)
            while True:
                try:
                    stream.write(block)
                except OSError as error:
                    if error.errno != errno.ENOSPC:
                        raise
                    break
    else:
        filler.mkdir()
        for index in range(10000):
            try:
                (filler / str(index)).touch(exist_ok=False)
            except OSError as error:
                if error.errno != errno.ENOSPC:
                    raise
                break
        else:
            raise RuntimeError("Inode exhaustion did not occur")
    return filler


def remove_filler(path):
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def io_points(operation, fts):
    points = [
        ("first", "sync" if operation == "flush" else "write", "", 1),
        ("partial-write", "write", "", 2),
        ("manifest", "write", "/manifest.", 1),
    ]
    if operation == "optimize":
        points += [("forward", "write", ".ipc", 2), ("vector", "write", ".proxima", 1)]
        if fts:
            points.append(("fts", "write", "fts", 1))
    return points


def fault_case(
    args, folder, fts, operation, fault, once=False, path_filter="", after=1
):
    with Volume(folder) as volume:
        database = volume.mountpoint / "collection"
        run_worker(args, database, "prepare", fts, folder)
        audit = folder / "audit.log"
        arm = folder / "armed"
        env = {
            "LD_PRELOAD": str(args.injector),
            "ZVEC_FAULT_ROOT": str(database),
            "ZVEC_FAULT_AUDIT": str(audit),
            "ZVEC_FAULT_ARM": str(arm),
        }
        if args.mode == "io-errors":
            env.update(
                ZVEC_FAULT_OPERATION=fault,
                ZVEC_FAULT_AFTER=str(after),
                ZVEC_FAULT_PATH=path_filter,
            )
            (folder / "fault.json").write_text(
                json.dumps(
                    {
                        "operation": fault,
                        "path_filter": path_filter,
                        "after": after,
                        "once": once,
                    },
                    indent=2,
                )
            )
            if once:
                env["ZVEC_FAULT_ONCE"] = "1"
        filler = None
        with paused_worker(args, database, operation, fts, folder, env) as process:
            if args.mode == "disk-full":
                filler = fill(volume, fault)
            else:
                arm.touch()
            code = resume(process)
        evidence = audit.read_text() if audit.exists() else ""
        marker = f"errno={errno.ENOSPC} " if args.mode == "disk-full" else "INJECT "
        problems = []
        if marker not in evidence:
            problems.append("Required storage fault was not observed in the worker")
        if args.mode == "io-errors":
            injections = [
                line for line in evidence.splitlines() if line.startswith("INJECT ")
            ]
            if not injections or any(path_filter not in line for line in injections):
                problems.append("Injection did not hit the requested file category")
            if after > 1 and "PASS " not in evidence.split("INJECT ", 1)[0]:
                problems.append("No successful write before the delayed injection")
        # A swallowed storage failure must not be treated as a successful test.
        if code != 2 or "STORAGE_ERROR:" not in (folder / "operation.log").read_text():
            problems.append(f"Expected explicit API failure, worker exited with {code}")
        if filler:
            remove_filler(filler)
        if arm.exists():
            arm.unlink()
        # Save the failed-operation state before recovery modifies the collection.
        volume.unmount()
        shutil.copyfile(volume.image, folder / "after-fault.img")
        volume.mount(volume.device)
        # Verify recovery even when the original operation incorrectly succeeded.
        try:
            run_worker(
                args,
                database,
                "verify",
                fts,
                folder,
                minimum=128 if operation == "optimize" else 64,
            )
        except Exception as error:
            problems.append(str(error))
        if problems:
            raise RuntimeError(
                "\n".join(problems) + "\n" + (folder / "operation.log").read_text()
            )


def read_log(path):
    """Read the documented little-endian dm-log-writes v1 record headers."""
    with path.open("rb") as stream:
        header = stream.read(28)
        magic, version, count, sector_size = struct.unpack("<QQQI", header)
        if magic != 0x6A736677736872 or version != 1 or sector_size not in (512, 4096):
            raise ValueError("Unsupported or corrupt dm-log-writes header")
        stream.seek(sector_size)
        entries = []
        for index in range(count):
            record = stream.read(sector_size)
            if len(record) != sector_size:
                raise ValueError("Truncated log record")
            _, sectors, flags, data_len = struct.unpack_from("<QQQQ", record)
            mark = None
            if flags & 8:
                if data_len > sector_size - 32:
                    raise ValueError("Oversized log mark")
                mark = record[32 : 32 + data_len].rstrip(b"\0").decode()
            entries.append((index, flags, mark))
            if not flags & 4:
                stream.seek(sectors * sector_size, os.SEEK_CUR)
        return entries


def choose_cuts(entries, count, seed):
    marks = {mark: index for index, _, mark in entries if mark}
    begin, end = marks["operation-begin"], marks["operation-durable"]
    if end <= begin + 1:
        raise RuntimeError("No recorded operation writes between marks")
    # Include both sides of FLUSH/FUA boundaries, plus seeded write cut points.
    candidates = set()
    for index, flags, _ in entries:
        if begin < index < end and flags & 3:
            candidates.update((index, index + 1))
    rng = random.Random(seed)
    chosen = {begin + 1, end + 1}
    boundary = sorted(candidates - chosen)
    chosen.update(rng.sample(boundary, min(len(boundary), max(0, count - 2))))
    others = sorted(set(range(begin + 1, end + 1)) - chosen)
    chosen.update(rng.sample(others, min(len(others), max(0, count - len(chosen)))))
    return sorted(chosen), end + 1


def power_case(args, folder, fts, operation):
    with Volume(folder) as volume:
        database = volume.mountpoint / "collection"
        run_worker(args, database, "prepare", fts, folder)
        volume.unmount()
        baseline = folder / "baseline.img"
        shutil.copyfile(volume.image, baseline)
        log_image = folder / "writes.img"
        with log_image.open("wb") as stream:
            stream.truncate(256 * 1024 * 1024)
        log_device = volume.loop(log_image)
        volume.mapper = "zvec-stability-" + uuid.uuid4().hex[:12]
        sectors = volume.root("blockdev", "--getsz", volume.device)
        volume.root(
            "dmsetup",
            "create",
            volume.mapper,
            "--table",
            f"0 {sectors} log-writes {volume.device} {log_device}",
        )
        volume.mount("/dev/mapper/" + volume.mapper)
        with paused_worker(args, database, operation, fts, folder) as process:
            volume.root(
                "dmsetup", "message", volume.mapper, "0", "mark operation-begin"
            )
            code = resume(process)
            if code != 0 or "DURABLE\n" not in (folder / "operation.log").read_text():
                raise RuntimeError(
                    "Recording workload failed:\n"
                    + (folder / "operation.log").read_text()
                )
            volume.root(
                "dmsetup", "message", volume.mapper, "0", "mark operation-durable"
            )
        # Unmount completes recording, but replay excludes its later writes.
        volume.unmount()
        volume.root("dmsetup", "remove", volume.mapper)
        volume.mapper = None
        entries = read_log(log_image)
        cuts, durable = choose_cuts(entries, args.cuts, args.seed)
        (folder / "cuts.json").write_text(
            json.dumps(
                {"seed": args.seed, "cuts": cuts, "durable_limit": durable}, indent=2
            )
        )
        replay = folder / "replay.img"
        replay_device = None
        problems = []
        for limit in cuts:
            detail = folder / f"cut-{limit}"
            detail.mkdir()
            # Always reset to the same baseline; mounting and validation mutate it.
            if replay_device:
                volume.root("losetup", "--detach", replay_device)
                volume.loops.remove(replay_device)
            shutil.copyfile(baseline, replay)
            replay_device = volume.loop(replay)
            try:
                volume.root(
                    args.replay_log,
                    "--log",
                    log_image,
                    "--replay",
                    replay_device,
                    "--limit",
                    str(limit),
                )
                volume.mount(replay_device)
                run_worker(
                    args,
                    database,
                    "verify",
                    fts,
                    detail,
                    minimum=128 if operation == "optimize" or limit >= durable else 64,
                )
            except Exception:
                error = traceback.format_exc()
                (detail / "failure.txt").write_text(error)
                problems.append(f"cut={limit}: {error}")
            finally:
                volume.unmount()
            # Baseline + writes.img + cut number reconstruct every failed image.
        if problems:
            raise RuntimeError("\n".join(problems))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True, choices=("disk-full", "io-errors", "power-loss")
    )
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--injector", type=Path)
    parser.add_argument("--replay-log", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=759)
    parser.add_argument("--cuts", type=int, default=8)
    args = parser.parse_args()

    # Let context managers unmount/detach owned resources on workflow cancellation.
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    args.output = args.output.resolve()
    args.worker = args.worker.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.injector:
        args.injector = args.injector.resolve()
    if args.replay_log:
        args.replay_log = args.replay_log.resolve()
    suite = Suite(args.output)
    (args.output / "configuration.json").write_text(
        json.dumps(
            {
                **{key: str(value) for key, value in vars(args).items()},
                "commit": os.environ.get("GITHUB_SHA"),
                "run_id": os.environ.get("GITHUB_RUN_ID"),
            },
            indent=2,
        )
    )

    def preflight(folder):
        if sys.platform != "linux" or os.getuid() == 0:
            raise RuntimeError(
                "Run as an unprivileged Linux user with passwordless sudo"
            )
        if not args.worker.is_file() or args.cuts < 2:
            raise RuntimeError("Missing worker or fewer than two power-loss cuts")
        command("sudo", "-n", "true", log=folder / "environment.log")
        command("uname", "-a", log=folder / "environment.log")
        for binary in ("losetup", "mkfs.ext4", "mount", "umount"):
            if not shutil.which(binary):
                raise RuntimeError(f"Missing dependency: {binary}")
        if args.mode == "power-loss":
            if not args.replay_log or not args.replay_log.is_file():
                raise RuntimeError("Missing pinned replay-log executable")
            command(
                "sudo",
                "-n",
                "modprobe",
                "dm_log_writes",
                log=folder / "environment.log",
            )
        elif not args.injector or not args.injector.is_file():
            raise RuntimeError("Missing test-only I/O audit/injection library")

    suite.case("preflight", preflight)
    if suite.results[-1][2]:
        return 1
    for fts in (False, True):
        kind = "fts" if fts else "plain"
        for operation in ("insert", "flush", "optimize"):
            if args.mode == "power-loss":
                suite.case(
                    f"{kind}-{operation}",
                    partial(power_case, args, fts=fts, operation=operation),
                )
            elif args.mode == "disk-full":
                for fault in ("blocks", "inodes"):
                    suite.case(
                        f"{kind}-{operation}-{fault}",
                        partial(
                            fault_case, args, fts=fts, operation=operation, fault=fault
                        ),
                    )
            else:
                for point, fault, path_filter, after in io_points(operation, fts):
                    for once in (True, False):
                        suite.case(
                            f"{kind}-{operation}-{point}-{'once' if once else 'persistent'}",
                            partial(
                                fault_case,
                                args,
                                fts=fts,
                                operation=operation,
                                fault=fault,
                                once=once,
                                path_filter=path_filter,
                                after=after,
                            ),
                        )
    return int(any(result[2] for result in suite.results))


if __name__ == "__main__":
    sys.exit(main())
