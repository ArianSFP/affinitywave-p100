#!/usr/bin/env python3
"""Isolated four-P100 jobs, with a lock and watchdog scoped to our child."""
import argparse
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
CUDA = Path('/usr/local/cuda-12.8')
MODEL = Path('/home/arian/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-Q8_0.gguf')


def fingerprint(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def environment(build, tokens, variant):
    docs = ROOT / 'docs/p100-pairwave-project'
    env = dict(HOME='/home/arian', PATH=f'{CUDA}/bin:/usr/bin:/bin',
               LD_LIBRARY_PATH=f'{build}/bin:{CUDA}/lib64:/usr/lib/x86_64-linux-gnu',
               CUDA_VISIBLE_DEVICES='0,1,2,3')
    for name in ['NCCL_IB_DISABLE', 'NCCL_CUMEM_ENABLE']:
        env[name] = '1' if name == 'NCCL_IB_DISABLE' else '0'
    env['NCCL_ALGO'] = 'Ring'
    for suffix in ['P2P', 'GRAPHS_PRE_AMPERE', 'GRAPHS_SPLIT_BUFFER',
                   'Q8_1_DEDUP', 'FORCE_CUBLAS_COMPUTE_32F', 'MOE_CSORT',
                   'MOE_EP', 'MOE_CSORT_REUSE', 'MOE_F16GATHER', 'MOE_PINNED',
                   'MOE_GROUPED', 'MOE_PLAN']:
        env['GGML_CUDA_' + suffix] = '1'
    env['GGML_META_SUBMIT_THREADS'] = '1'
    env['GGML_CUDA_MOE_EPLB_MAP'] = str(docs / 'evidence/runtime/placement-primary.eplb')
    if variant != 'normal':
        env['GGML_CUDA_AFFINITY_WAVE'] = '1'
        settings = dict(MAP=str(docs / 'evidence/runtime/placement-hot16.json'),
            WIRE='f32', PARTIAL='bf16', NCCL_SUM='0', NCCL_ORDER_SUM='1', CHECK='1',
            Q8_LAYOUT='t64k32', Q8_KERNEL='interleave', Q8_ENGINE='cohortrail',
            COHORTRAIL_P2_CTAS='3', COHORTRAIL_SINGLE_CTAS='3', SERVICE='legacy',
            DIAGONAL_SERVICE='panel2048', ROUTE_SCATTER='deterministic',
            DIRECT_OWNER='0', DENSE_T64='0', DENSE_SELECTORS='exact', DOWN_CACHE='0',
            M64_SPLIT='2', HOME='layer', DECODE='t64', WAVE_DRY='0', WAVE_DENSE='1',
            WAVE_TOKEN_SPLIT='1', WAVE_TOKEN_PLAN='0', WAVE_DENSE_BENCH='service',
            WAVE_OUTPUT='1', LANE_STAGGER='1', CORRIDOR_EARLY='all',
            CORRIDOR_STATE_SPLIT='1', PRECAPTURE='1', PRECAPTURE_OUTPUT='1',
            SHARED_SERVICE='0', LANE_BALANCE='0', DEBUG_NODE_SYNC='0',
            DEBUG_LIVE_SYNC='0', GROUP_CELLS='1', GROUP_PATTERN='1111',
            GDN_CHUNKED='2', GDN_WARPS='4', GDN_WY_GRAPH='0', FUSED_GATE_UP='0',
            FA_QUERY_TILE='4', WAVE_TOKENS=str(tokens), P100_EXACT='1',
            P100_EXACT_SUM_WIDTH='4', PAIRWAVE_SERVICE='0', HEADFOLD='0')
        if variant == 'serve':
            settings['SERVE'] = '1'
        env.update({'GGML_CUDA_AW_' + k: v for k, v in settings.items()})
    return env


def stop(child, timeout=10):
    if child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def main():
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--tokens', type=int, default=8128)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--variant', choices=['qualified', 'serve', 'normal'], default='qualified')
    parser.add_argument('--mode', choices=['bench', 'ppl', 'kld', 'server', 'serve', 'validate', 'quality'], default='bench')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--ctx', type=int, default=16384)
    parser.add_argument('--port', type=int, default=18097)
    parser.add_argument('--host', default='127.0.0.1',
                        help='server bind address; use 0.0.0.0 for LAN access')
    parser.add_argument('--webui', action='store_true',
                        help='enable the embedded llama.cpp Web UI')
    parser.add_argument('--suite', choices=['smoke', 'matrix', 'short', 'soak'], default='smoke')
    parser.add_argument('--prewarm', default='',
                        help='comma-separated fresh prompt sizes to prewarm before probing')
    parser.add_argument('--env', action='append', default=[], metavar='NAME=VALUE')
    parser.add_argument('--debugger', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--verbose', action='store_true',
                        help='stream server logs to the terminal and enable llama-server -v')
    parser.add_argument('--warmup', action='store_true',
                        help='allow llama-server startup warmup')
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--ubatch', type=int)
    args = parser.parse_args()
    prewarm = tuple(int(value) for value in args.prewarm.split(',') if value.strip())
    if any(value < 1 for value in prewarm):
        parser.error('--prewarm sizes must be positive')
    if not args.tag or Path(args.tag).name != args.tag or args.tokens < 1 or args.repeats < 1:
        parser.error('invalid tag, token count, or repetitions')
    if args.timeout < 0 or (args.timeout == 0 and args.mode != 'serve'):
        parser.error('timeout must be positive, except persistent --mode serve --timeout 0')
    if args.profile and args.debugger:
        parser.error('choose profiling or debugging, not both')
    build = args.build.resolve()
    target = ROOT / 'results/coding-serve' / args.tag
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = open('/tmp/affinitywave-4gpu.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    clients = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                      '--format=csv,noheader'], text=True).strip()
    if clients:
        raise RuntimeError('existing GPU compute clients: ' + clients)
    if os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize < 6 * 1024**3:
        raise RuntimeError('less than 6 GiB free before job')
    env = environment(build, args.tokens, args.variant)
    for setting in args.env:
        name, separator, value = setting.partition('=')
        if not separator or not name.startswith(('GGML_CUDA_AW_', 'GGML_META_')):
            parser.error('--env only accepts explicit AW or META experiment settings')
        env[name] = value
    executable = {'bench': 'llama-bench', 'ppl': 'llama-perplexity',
                  'kld': 'llama-perplexity', 'server': 'llama-server', 'serve': 'llama-server',
                  'validate': 'test-aw-serving', 'quality': 'test-aw-serving'}[args.mode]
    cmd = ['taskset', '--cpu-list', '0-11', str(build / 'bin' / executable),
           '-m', str(MODEL), '-ngl', '99', '-sm', 'tensor', '-fa', '1',
           '-b', str(args.tokens), '-ub', str(args.ubatch or args.tokens)]
    if args.mode == 'bench':
        cmd += ['-mmp', '0', '-p', str(args.tokens), '-n', '0', '-r', str(args.repeats), '-o', 'json']
    elif args.mode in ['ppl', 'kld']:
        cmd += ['--no-mmap', '--no-warmup', '-f',
                '/home/arian/llama.cpp-q36-decodeopt/wikitext-2-raw/wiki.test.raw',
                '-c', str(args.tokens), '--chunks', '1']
        if args.mode == 'kld':
            if args.reference is None or not args.reference.is_file():
                parser.error('KLD mode requires an existing --reference logits file')
            cmd += ['--kl-divergence', '--kl-divergence-base', str(args.reference.resolve())]
        else:
            cmd += ['--save-all-logits', str(target) + '.logits']
    elif args.mode in ['validate', 'quality']:
        validation_output = Path(str(target) + ('.quality' if args.mode == 'quality' else '.boundaries'))
        if validation_output.exists():
            raise RuntimeError('validation output already exists')
        env['GGML_CUDA_AW_VALIDATION_OUTPUT'] = str(validation_output)
        cmd += ['-c', str(args.ctx), '-np', '1', '-t', '12', '-tb', '12',
                '--no-warmup', '--no-mmap']
        if args.mode == 'quality':
            env['GGML_CUDA_AW_VALIDATION_TOKENS'] = str(args.tokens)
            cmd += ['-f', '/home/arian/llama.cpp-q36-decodeopt/wikitext-2-raw/wiki.test.raw']
    else:
        if args.variant == 'qualified':
            parser.error('the fixed-shape qualified path is not a server configuration')
        cmd += ['-c', str(args.ctx), '-np', '1', '-t', '12', '-tb', '12',
                '--no-cont-batching', '--cache-prompt', '--no-mmap',
                '--host', args.host, '--port', str(args.port),
                '--webui' if args.webui else '--no-webui', '--jinja']
        if args.verbose:
            cmd.append('--verbose')
        if not args.warmup:
            cmd.insert(cmd.index('--no-mmap'), '--no-warmup')
    with open(str(target) + '.meta.json', 'x') as out:
        binaries = [build / 'bin' / name for name in
                    [executable, 'libllama.so', 'libggml-base.so', 'libggml-cuda.so']]
        json.dump(dict(command=cmd, environment=env, started=time.time(),
                       wrapper=dict(profile=args.profile, debugger=args.debugger),
                       binary_sha256={str(path): fingerprint(path) for path in binaries},
                       source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                       source_diff_sha256=hashlib.sha256(subprocess.check_output(
                           ['git', 'diff', '--binary'], cwd=ROOT)).hexdigest()), out, indent=2)
    if args.debugger:
        cmd = ['gdb', '--batch', '-ex', 'set pagination off', '-ex', 'set debuginfod enabled off',
               '-ex', 'set print thread-events off', '-ex', 'run', '-ex', 'bt 24',
               '-ex', 'thread apply all bt 4', '--args'] + cmd
    if args.profile:
        # Set the clean environment on nsys itself, never on its injected child.
        cmd = [str(CUDA / 'bin/nsys'), 'profile', '--sample=none', '--cpuctxsw=none',
               '--trace=cuda,nvtx,osrt', '--cuda-graph-trace=node', '--delay=80',
               '--duration=60', '--kill=none', '--stop-on-exit=true',
               '--output=' + str(target)] + cmd
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    child = None
    executor = None
    probe = None
    probe_stop = threading.Event()
    deadline = time.monotonic() + args.timeout if args.timeout else None
    xs = Path('/home/arian/.xsession-errors')
    try:
        with ExitStack() as logs:
            if args.verbose:
                stdout = stderr = None
            else:
                stdout = logs.enter_context(open(str(target) + '.out', 'x'))
                stderr = logs.enter_context(open(str(target) + '.err', 'x'))
            child = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            print(f'job pid={child.pid} tag={args.tag}', flush=True)
            if args.mode == 'server' or (args.mode == 'serve' and prewarm):
                from server_probe import prewarm_server, run_probe
                executor = ThreadPoolExecutor(max_workers=1)
                if args.mode == 'server':
                    probe = executor.submit(
                        run_probe, args.port, args.suite, target, args.ctx,
                        probe_stop, prewarm)
                else:
                    probe = executor.submit(
                        prewarm_server, args.port, prewarm, args.ctx, probe_stop)
            while child.poll() is None:
                if probe is not None and probe.done():
                    probe.result()
                    if args.mode == 'server':
                        print(f'server probe PASS tag={args.tag}', flush=True)
                        return
                    print(f'server prewarm PASS tag={args.tag}', flush=True)
                    probe = None
                free = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
                if xs.exists() and xs.stat().st_size > 50 * 1024**2:
                    with xs.open('r+') as log:
                        log.truncate(0)
                    raise RuntimeError('xsession watchdog tripped; log truncated, stopping our job')
                if free < 5 * 1024**3 or (deadline is not None and time.monotonic() > deadline):
                    raise RuntimeError('disk/deadline watchdog tripped; stopping our job')
                time.sleep(1)
            if probe is not None:
                raise RuntimeError(f'server exited before probe completion: {child.returncode}')
            print(f'job exit={child.returncode} tag={args.tag}', flush=True)
            if child.returncode:
                raise SystemExit(child.returncode)
    finally:
        probe_stop.set()
        if child is not None:
            stop(child, timeout=30 if args.profile else 10)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        lock.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
