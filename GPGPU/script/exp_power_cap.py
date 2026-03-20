import os
import subprocess
import time
import signal
import argparse
import csv
import re
import sys

# Add coSched to path so we can import shared config
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'coSched'))
from config import (
    ML_MIN_PER_GPU_CAP, ML_MAX_PER_GPU_CAP,
    ML_BATCH_SIZE, ML_BATCH_SIZE_OVERRIDE,
    ML_EPOCHS, ML_LR, SYSTEM,
)

num_gpu = 4
system = SYSTEM

# System-dependent benchmark paths
SYSTEM_CONFIG = {
    "V100": {
        "spec_benchmark_root": os.path.expanduser("~/benchmark/spec-V100"),
        "cuda_build_root": "build-sm70",
    },
    "H100": {
        "spec_benchmark_root": os.path.expanduser("~/benchmark/spec-H100"),
        "cuda_build_root": "build-sm90",
    },
    "A100": {
        "spec_benchmark_root": os.path.expanduser("~/benchmark/spec-A100"),
        "cuda_build_root": "build-sm80",
    },
}

# Define paths and executables
home_dir = os.path.expanduser('~')
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set system-dependent env vars for benchmark scripts
_sys_cfg = SYSTEM_CONFIG.get(system, {})
os.environ["SPEC_BENCHMARK_ROOT"] = _sys_cfg.get("spec_benchmark_root", "")
os.environ["BENCHMARK_BUILD_ROOT"] = os.path.join(
    script_dir, "run_benchmark", _sys_cfg.get("cuda_build_root", "build-sm90")
)
python_executable = os.path.join(home_dir, "env/ml/bin/python3")

# scripts for CPU, GPU power monitoring
read_cpu_power = os.path.join(script_dir, "power_util/read_cpu_power.py")
read_gpu_power = os.path.join(script_dir, "power_util/read_gpu_power.py")
read_gpu_metrics = os.path.join(script_dir, "power_util/read_gpu_metrics.py")
read_cpu_ips = os.path.join(script_dir, "power_util/read_cpu_ips.py")
read_mem = os.path.join(script_dir, "power_util/read_mem.py")
read_cpu_metrics = os.path.join(script_dir, "power_util/read_cpu_metrics.py")

# scritps for running various benchmarks
run_app = os.path.join(script_dir, "run_benchmark/run_app.py")


hec_benchmarks = ["addBiasResidualLayerNorm", "aobench", "background-subtract", "chacha20", "convolution3D", "dropout", "extrema", "fft", "kalman", "knn", "softmax", "stencil3d", "zmddft", "zoom"]
altis_benchmarks_0 = ["maxflops"]
altis_benchmarks_1 = ['bfs','gemm','gups','pathfinder','sort']
altis_benchmarks_2 = ['cfd','cfd_double','fdtd2d','kmeans','lavamd',
                      'nw','particlefilter_float','particlefilter_naive','raytracing',
                      'srad','where']
ecp_benchmarks = ['XSBench','miniGAN','CRADL','sw4lite','Laghos','bert','UNet', "gpt2",'Resnet50','lammps','gromacs',"NAMD"]

ecp_benchmarks = ["gpt2","bert"]

spec_benchmarks = ['lbm', 'cloverleaf', 'tealeaf', 'minisweep', 'pot3d', 'miniweather', 'hpgmg']

spec_benchmarks = ['hpgmg']

cuda_benchmarks = ['conjugateGradientMultiDeviceCG','MonteCarloMultiGPU','simpleCUBLASXT',
                   'simpleCUFFT_MGPU', 'simpleCUFFT_2d_MGPU','simpleMultiGPU','simpleP2P',
                   'streamOrderedAllocationP2P']




ml_models = ["resnet101","resnet152", "vgg19","vgg16","resnet50"]
ml_models = ["resnet152"]

cpu_caps = [700]
GPU_ct = [1,2,3,4]
# GPU_ct = [2]
gpu_caps = [2800]



def _upsert_runtime_row(runtime_csv_path, power_cap, gpu_count, runtime_seconds):
    """
    Upsert runtime.csv by unique key (power_cap, gpu_count):
      - update runtime_seconds if key exists
      - append new row if key does not exist
    """
    rows = []
    updated = False

    if os.path.exists(runtime_csv_path) and os.path.getsize(runtime_csv_path) > 0:
        with open(runtime_csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cap = float(row.get("power_cap"))
                gct = float(row.get("gpu_count"))
                if cap is None or gct is None:
                    continue
                if int(round(cap)) == int(power_cap) and int(round(gct)) == int(gpu_count):
                    row["runtime_seconds"] = f"{runtime_seconds}"
                    updated = True
                rows.append(row)

    if not updated:
        rows.append({
            "power_cap": str(power_cap),
            "gpu_count": str(gpu_count),
            "runtime_seconds": f"{runtime_seconds}",
        })

    with open(runtime_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["power_cap", "gpu_count", "runtime_seconds"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "power_cap": row.get("power_cap", ""),
                "gpu_count": row.get("gpu_count", ""),
                "runtime_seconds": row.get("runtime_seconds", ""),
            })


def _upsert_csv_row(csv_path, fieldnames, key_fields, row_values):
    """
    Generic CSV upsert by key_fields:
      - update row if key exists
      - append row if key does not exist
    """
    rows = []
    updated = False

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                is_match = True
                for k in key_fields:
                    if str(row.get(k, "")) != str(row_values.get(k, "")):
                        is_match = False
                        break
                if is_match:
                    for fn in fieldnames:
                        row[fn] = str(row_values.get(fn, row.get(fn, "")))
                    updated = True
                rows.append(row)

    if not updated:
        rows.append({fn: str(row_values.get(fn, "")) for fn in fieldnames})

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({fn: row.get(fn, "") for fn in fieldnames})


def _select_ml_python():
    return python_executable


def _extract_avg_train_throughput(stdout_text):
    # Preferred metric from dl.py summary
    m = re.search(
        r"Average train throughput \(excluding epoch 1\):\s*([0-9]+(?:\.[0-9]+)?)\s*images/sec",
        stdout_text,
    )
    if m:
        return float(m.group(1))

    # Fallback: use last epoch throughput line if summary is unavailable.
    fallback = re.findall(
        r"Epoch\s+\d+\s+Complete\s*-\s*Throughput:\s*([0-9]+(?:\.[0-9]+)?)\s*images/sec",
        stdout_text,
    )
    if fallback:
        return float(fallback[-1])
    return None


def _extract_token_throughput(stdout_text):
    patterns = [
        r"([0-9]+(?:\.[0-9]+)?)\s*tokens/sec",
        r"([0-9]+(?:\.[0-9]+)?)\s*token/sec",
        r"([0-9]+(?:\.[0-9]+)?)\s*tokens/s",
        r"([0-9]+(?:\.[0-9]+)?)\s*token/s",
        r"tokens/sec\s*([0-9]+(?:\.[0-9]+)?)",
        r"token/sec\s*([0-9]+(?:\.[0-9]+)?)",
        r"tokens/s\s*([0-9]+(?:\.[0-9]+)?)",
        r"token/s\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for pat in patterns:
        matches = re.findall(pat, stdout_text, flags=re.IGNORECASE)
        if matches:
            return float(matches[-1])
    return None


def _is_throughput_benchmark(suite_name, benchmark_name):
    if str(suite_name).lower() != "ecp":
        return False
    bn = str(benchmark_name).lower()
    return "bert" in bn or "gpt2" in bn


def _gpu_counts_for_benchmark(suite_name, benchmark_name):
    if str(suite_name).lower() == "cuda":
        bn = str(benchmark_name)
        if bn == "simpleCUFFT_MGPU":
            return sorted(set([1, 2, 4]) & set(GPU_ct))
        if bn in ("simpleP2P", "streamOrderedAllocationP2P"):
            return sorted(set([2]) & set(GPU_ct))
        return sorted(set([1, 2, 3, 4]) & set(GPU_ct))
    return GPU_ct


def _per_gpu_cap_from_total(total_gpu_cap, gpu_count):
    if gpu_count <= 0:
        return None
    per_gpu_cap = float(total_gpu_cap) / float(gpu_count)
    if per_gpu_cap < ML_MIN_PER_GPU_CAP:
        return None
    # If total budget is above per-GPU TDP, still run with fewer GPUs by capping at TDP.
    per_gpu_cap = min(per_gpu_cap, ML_MAX_PER_GPU_CAP)
    return int(per_gpu_cap)


def _set_power_cap(cpu_cap, per_gpu_cap):
    subprocess.run(
        [os.path.join(script_dir, "power_util/cap.sh"), str(cpu_cap), str(per_gpu_cap)],
        check=True,
    )
    time.sleep(0.2)


def _run_with_gpu_monitor(cmd, cwd, output_gpu_metrics, num_gpu_to_monitor):
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )

    monitor_cmd = [
        python_executable,
        read_gpu_metrics,
        "--output_csv",
        output_gpu_metrics,
        "--pid",
        str(process.pid),
        "--num_gpu",
        str(num_gpu_to_monitor),
    ]
    monitor_process = subprocess.Popen(
        monitor_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    output_lines = []
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
        output_lines.append(line)
    process.stdout.close()
    return_code = process.wait()

    monitor_stdout = ""
    monitor_stderr = ""
    try:
        # Give monitor some time to flush/write final samples after app exits.
        monitor_stdout, monitor_stderr = monitor_process.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            monitor_process.terminate()
            monitor_stdout, monitor_stderr = monitor_process.communicate(timeout=5)
        except Exception:
            pass
    except Exception:
        pass

    return return_code, "".join(output_lines), monitor_stdout, monitor_stderr



def run_ml_experiment(model_name=None):
    cpu_cap = 700
    ml_dir = os.path.join(home_dir, "power", "ML")
    ml_script = os.path.join(ml_dir, "dl.py")
    ml_python = _select_ml_python()

    print("[ML] Using python: {}".format(ml_python))

    if model_name:
        if model_name not in ml_models:
            raise ValueError(f"Unknown ML model '{model_name}'. Available: {ml_models}")
        models = [model_name]
    else:
        models = ml_models

    output_root_dir = os.path.abspath(
        os.path.join(script_dir, "..", "data", system, "ml_power_motif")
    )
    os.makedirs(output_root_dir, exist_ok=True)
    throughput_csv_by_model = {}
    for model in models:
        model_output_dir = os.path.join(output_root_dir, model)
        os.makedirs(model_output_dir, exist_ok=True)
        output_csv = os.path.join(model_output_dir, "throughput.csv")
        throughput_csv_by_model[model] = output_csv
        if (not os.path.exists(output_csv)) or os.path.getsize(output_csv) == 0:
            with open(output_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "total_gpu_cap",
                    "gpu_count",
                    "per_gpu_cap",
                    "model",
                    "throughput_images_per_sec",
                    "status",
                ])

    for model in models:
        gpu_counts = GPU_ct

        for g_cnt in gpu_counts:
            for total_gpu_cap in gpu_caps:
                per_gpu_cap_int = _per_gpu_cap_from_total(total_gpu_cap, g_cnt)
                if per_gpu_cap_int is None:
                    # Skip invalid combinations that violate H100 per-GPU cap range.
                    continue

                # _set_power_cap(cpu_cap, per_gpu_cap_int)

                cmd = [
                    ml_python,
                    ml_script,
                    "--model", model,
                    "--num-gpus", str(g_cnt),
                    "--batch-size", str(ML_BATCH_SIZE_OVERRIDE.get(system, {}).get(model, ML_BATCH_SIZE)),
                    "--epochs", str(ML_EPOCHS),
                    "--lr", str(ML_LR),
                ]

                print(
                    f"[ML] Running model={model} total_cap={total_gpu_cap} "
                    f"gpus={g_cnt} per_gpu_cap={per_gpu_cap_int}"
                )
                model_output_dir = os.path.join(output_root_dir, model)
                output_gpu_metrics = os.path.join(
                    model_output_dir,
                    f"{total_gpu_cap}_{g_cnt}_gpu_metrics.csv",
                )
                return_code, run_output, _, monitor_stderr = _run_with_gpu_monitor(
                    cmd=cmd,
                    cwd=ml_dir,
                    output_gpu_metrics=output_gpu_metrics,
                    num_gpu_to_monitor=g_cnt,
                )
                throughput = _extract_avg_train_throughput(run_output)
                status = "ok" if (return_code == 0 and throughput is not None) else "failed"

                _upsert_csv_row(
                    csv_path=throughput_csv_by_model[model],
                    fieldnames=[
                        "total_gpu_cap",
                        "gpu_count",
                        "per_gpu_cap",
                        "model",
                        "throughput_images_per_sec",
                        "status",
                    ],
                    key_fields=["total_gpu_cap", "gpu_count"],
                    row_values={
                        "total_gpu_cap": total_gpu_cap,
                        "gpu_count": g_cnt,
                        "per_gpu_cap": per_gpu_cap_int,
                        "model": model,
                        "throughput_images_per_sec": f"{throughput:.2f}" if throughput is not None else "",
                        "status": status,
                    },
                )

                if status != "ok":
                    print(
                        f"[ML][WARN] model={model} total_cap={total_gpu_cap} gpus={g_cnt} "
                        f"per_gpu_cap={per_gpu_cap_int} failed. returncode={return_code}"
                    )
                    if run_output:
                        print(run_output[-1000:])
                if (not os.path.exists(output_gpu_metrics)) or os.path.getsize(output_gpu_metrics) == 0:
                    print(
                        f"[ML][WARN] missing/empty GPU metrics file for model={model} "
                        f"cap={total_gpu_cap} gpus={g_cnt}: {output_gpu_metrics}"
                    )
                    if monitor_stderr:
                        print("[ML][MONITOR STDERR]")
                        print(monitor_stderr[-1000:])

    # subprocess.run([os.path.join(script_dir, "power_util/cap.sh"), str(700), str(700)], check=True)
    print("[ML] Throughput results saved to:")
    for model in models:
        print(f"  {throughput_csv_by_model[model]}")





def run_benchmark(benchmark_script_dir,benchmark, suite, test, size,cap_type):
    cpu_cap = 700
    is_bert = _is_throughput_benchmark(suite, benchmark)

    # For ECP BERT/GPT2, store both throughput and GPU metrics under ml_power_motif.
    if is_bert:
        output_dir = os.path.abspath(
            os.path.join(script_dir, "..", "data", system, "ml_power_motif", benchmark)
        )
    else:
        output_dir = f"../data/{system}/{suite}_power_motif/{benchmark}"
    os.makedirs(output_dir, exist_ok=True)
    output_runtime = f"{output_dir}/runtime.csv"

    bert_throughput_csv = None
    if is_bert:
        bert_throughput_csv = os.path.join(output_dir, "throughput.csv")
        if (not os.path.exists(bert_throughput_csv)) or os.path.getsize(bert_throughput_csv) == 0:
            with open(bert_throughput_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "total_gpu_cap",
                    "gpu_count",
                    "per_gpu_cap",
                    "benchmark",
                    "throughput_tokens_per_sec",
                    "status",
                ])

    gpu_counts = _gpu_counts_for_benchmark(suite, benchmark)

    for g_cnt in gpu_counts:
        for total_gpu_cap in gpu_caps:
            per_gpu_cap = _per_gpu_cap_from_total(total_gpu_cap, g_cnt)
            if per_gpu_cap is None:
                continue

            output_gpu_metrics = f"{output_dir}/{total_gpu_cap}_{g_cnt}_gpu_metrics.csv"
            # _set_power_cap(cpu_cap, per_gpu_cap)
            run_benchmark_command = [
                "bash",
                os.path.join(home_dir, benchmark_script_dir, f"{benchmark}.sh"),
                str(g_cnt),
            ]

            print(
                f"[{suite.upper()}] Running benchmark={benchmark} total_cap={total_gpu_cap} "
                f"gpus={g_cnt} per_gpu_cap={per_gpu_cap}"
            )
            start = time.time()
            return_code, run_output, _, monitor_stderr = _run_with_gpu_monitor(
                cmd=run_benchmark_command,
                cwd=None,
                output_gpu_metrics=output_gpu_metrics,
                num_gpu_to_monitor=g_cnt,
            )
            elapsed_time = time.time() - start

            # Upsert runtime by (power_cap, gpu_count) for non-BERT benchmarks.
            if not is_bert:
                _upsert_runtime_row(
                    runtime_csv_path=output_runtime,
                    power_cap=total_gpu_cap,
                    gpu_count=g_cnt,
                    runtime_seconds=elapsed_time,
                )

            if bert_throughput_csv is not None:
                tok_s = _extract_token_throughput(run_output)
                status = "ok" if (return_code == 0 and tok_s is not None) else "failed"
                _upsert_csv_row(
                    csv_path=bert_throughput_csv,
                    fieldnames=[
                        "total_gpu_cap",
                        "gpu_count",
                        "per_gpu_cap",
                        "benchmark",
                        "throughput_tokens_per_sec",
                        "status",
                    ],
                    key_fields=["total_gpu_cap", "gpu_count"],
                    row_values={
                        "total_gpu_cap": total_gpu_cap,
                        "gpu_count": g_cnt,
                        "per_gpu_cap": per_gpu_cap,
                        "benchmark": benchmark,
                        "throughput_tokens_per_sec": f"{tok_s:.2f}" if tok_s is not None else "",
                        "status": status,
                    },
                )

            if return_code != 0:
                print(
                    f"[{suite.upper()}][WARN] benchmark={benchmark} total_cap={total_gpu_cap} "
                    f"gpus={g_cnt} per_gpu_cap={per_gpu_cap} failed. returncode={return_code}"
                )
            if (not os.path.exists(output_gpu_metrics)) or os.path.getsize(output_gpu_metrics) == 0:
                print(
                    f"[{suite.upper()}][WARN] missing/empty GPU metrics file for benchmark={benchmark} "
                    f"cap={total_gpu_cap} gpus={g_cnt}: {output_gpu_metrics}"
                )
                if monitor_stderr:
                    print("[BENCHMARK][MONITOR STDERR]")
                    print(monitor_stderr[-1000:])
            if bert_throughput_csv is not None and tok_s is None:
                print(
                    f"[{suite.upper()}][WARN] tokens/sec not found in output for benchmark={benchmark} "
                    f"cap={total_gpu_cap} gpus={g_cnt}"
                )



    # subprocess.run([os.path.join(script_dir, "power_util/cap.sh"), str(700), str(700)], check=True)
    if bert_throughput_csv is not None:
        print(f"[ECP] BERT throughput saved to: {bert_throughput_csv}")


if __name__ == "__main__":

   # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Run benchmarks and monitor power consumption.')
    parser.add_argument('--benchmark', type=str, help='Optional name of the benchmark to run', default=None)
    parser.add_argument('--test', type=int, help='whether it is a test run', default=None)
    parser.add_argument('--suite', type=int, help='0 for ECP, 1 for ALTIS, 2 for ML, 3 for HEC, 4 for SPEC, 5 for ALL, 6 for CUDA', default=1)
    parser.add_argument('--benchmark_size', type=int, help='0 for big, 1 for small', default=0)
    parser.add_argument('--cap_type', type=int, help='0 for cpu, 1 for gpu, 2 for dual', default=2)
    parser.add_argument('--num_gpu', type=int, default=1)

    args = parser.parse_args()
    benchmark = args.benchmark
    test = args.test
    suite = args.suite
    benchmark_size = args.benchmark_size
    cap_type = args.cap_type
    # num_gpu = args.num_gpu


    if suite == 0 or suite ==5:
        benchmark_script_dir = f"power/GPGPU/script/run_benchmark/ecp_script"
        # single test
        if benchmark:
            run_benchmark(benchmark_script_dir, benchmark,"ecp",test,benchmark_size,cap_type)
        # run all ecp benchmarks
        else:
            for benchmark in ecp_benchmarks:
                run_benchmark(benchmark_script_dir, benchmark,"ecp",test,benchmark_size,cap_type)
    

    if suite == 1 or suite ==5:
        # Map of benchmarks to their paths
        benchmark_paths = {
            "level0": altis_benchmarks_0,
            "level1": altis_benchmarks_1,
            "level2": altis_benchmarks_2
        }
    
        if benchmark:
            # Find which level the input benchmark belongs to
            found = False
            for level, benchmarks in benchmark_paths.items():
                if benchmark in benchmarks:
                    benchmark_script_dir = f"power/GPGPU/script/run_benchmark/altis_script/{level}"
                    run_benchmark(benchmark_script_dir, benchmark,"altis",test,benchmark_size,cap_type)
                    found = True
                    break
        else:
            for benchmark in altis_benchmarks_0:
                if benchmark_size==0:
                    benchmark_script_dir = "power/GPGPU/script/run_benchmark/altis_script/level0"
                else:
                    benchmark_script_dir = "power/GPGPU/script/run_benchmark/altis_script/level0"
                run_benchmark(benchmark_script_dir, benchmark,"altis",test,benchmark_size,cap_type)
            
            
            for benchmark in altis_benchmarks_1:
                if benchmark_size==0:
                    benchmark_script_dir = "power/GPGPU/script/run_benchmark/altis_script/level1"
                else:
                    benchmark_script_dir = "power/GPGPU/script/run_benchmark/altis_script/level1"
                run_benchmark(benchmark_script_dir, benchmark,"altis",test,benchmark_size,cap_type)
            
            
            for benchmark in altis_benchmarks_2:
                if benchmark_size==0:
                    benchmark_script_dir = "power/GPGPU/script/run_benchmark/altis_script/level2"
                else:
                    benchmark_script_dir = "power/GPGPU/script/run_benchmark/altis_script/level2"
                run_benchmark(benchmark_script_dir, benchmark,"altis",test,benchmark_size,cap_type)


    if suite == 2 or suite == 5:
        run_ml_experiment(model_name=benchmark)


    if suite == 3 or suite == 5:
        benchmark_script_dir = f"power/GPGPU/script/run_benchmark/hec_script"
         # single test
        if benchmark:
            run_benchmark(benchmark_script_dir, benchmark,"hec",test,benchmark_size,cap_type)
        # run all ecp benchmarks
        else:
            for benchmark in hec_benchmarks:
                run_benchmark(benchmark_script_dir, benchmark,"hec",test,benchmark_size,cap_type)

    if suite == 4 or suite == 5:
        benchmark_script_dir = f"power/GPGPU/script/run_benchmark/spec_script"
        # single test
        if benchmark:
            run_benchmark(benchmark_script_dir, benchmark,"spec",test,benchmark_size,cap_type)
        # run all spec benchmarks
        else:
            for benchmark in spec_benchmarks:
                run_benchmark(benchmark_script_dir, benchmark,"spec",test,benchmark_size,cap_type)

    if suite == 6 or suite == 5:
        benchmark_script_dir = f"power/GPGPU/script/run_benchmark/cuda_script"
        # single test
        if benchmark:
            run_benchmark(benchmark_script_dir, benchmark,"cuda",test,benchmark_size,cap_type)
        # run listed CUDA benchmarks
        else:
            for benchmark in cuda_benchmarks:
                run_benchmark(benchmark_script_dir, benchmark,"cuda",test,benchmark_size,cap_type)
