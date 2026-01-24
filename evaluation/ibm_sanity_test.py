import argparse
import time
from pathlib import Path

from qiskit import QuantumCircuit, transpile


def _load_token(path: str) -> str:
    token_path = Path(path)
    if not token_path.exists():
        raise FileNotFoundError(f"Token file not found: {token_path}")
    token = token_path.read_text().strip()
    if not token:
        raise ValueError(f"Token file is empty: {token_path}")
    return token


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sanity test a real IBM backend using Sampler job mode."
    )
    parser.add_argument(
        "--token-file",
        default="IBM_token_illinois.key",
        help="Path to IBM token file.",
    )
    parser.add_argument(
        "--backend",
        default="ibm_torino",
        help="Backend name (e.g. ibm_torino or ibm_marrakesh).",
    )
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument(
        "--warn-sec",
        type=float,
        default=1.0,
        help="Warn if wall time exceeds this many seconds.",
    )
    args = parser.parse_args()

    token = _load_token(args.token_file)
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler  # type: ignore

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    backend = service.backend(args.backend)

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    tcirc = transpile(qc, backend=backend, optimization_level=1)

    sampler = Sampler(mode=backend)
    t0 = time.perf_counter()
    job = sampler.run([tcirc], shots=args.shots)
    result = job.result()
    elapsed = time.perf_counter() - t0

    try:
        job_id = job.job_id()
    except Exception:
        job_id = None

    print(f"job_id={job_id}")
    print(f"elapsed_sec={elapsed:.2f}")
    if elapsed > args.warn_sec:
        print(f"WARNING: elapsed_sec exceeds {args.warn_sec:.2f}s")
    print(f"result_type={type(result).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
