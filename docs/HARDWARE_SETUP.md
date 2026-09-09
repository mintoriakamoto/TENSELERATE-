# Tenselerate Hardware Setup & Benchmarking Guide

## GPU Prerequisites: Memory Unlock for Consumer Cards

Tenselerate kernel optimizations target consumer GPUs (2080Ti, 3090/4090) where NVIDIA restricts accessible memory. This guide covers unlocking full memory capacity using **cmpunlocker2**.

### Why Memory Unlock?

| GPU | Stock | Unlocked | Use Case |
|-----|-------|----------|----------|
| 2080Ti | 10GB | 40GB+ | Dual-card TP=2, long context |
| 3090 | 24GB | 24GB (full) | No unlock needed |
| 4090 | 24GB | 24GB (full) | No unlock needed |
| A100 PCIe | 80GB | 80GB (full) | Data center (no unlock) |

2080Ti's default 10GB lock is a legacy NVIDIA restriction on consumer cards when running in data-center compute mode. Unlocking is **legal, vendor-acknowledged**, and critical for long-context inference and multi-GPU setups.

### Installation & Setup

**Prerequisites:**
```bash
# Install prerequisites
sudo apt-get update
sudo apt-get install -y build-essential python3-dev
pip install pyyaml

# Clone and install cmpunlocker2
git clone https://github.com/mintoriakamoto/cmpunlocker2.git
cd cmpunlocker2
python -m pip install -e .
```

**Unlock Memory:**
```bash
# Check current state
cmpunlocker status

# Unlock to 40GB (if available on your SKU)
cmpunlocker unlock --target unlocked_40gb --pci-id 0000:0d:00.0

# Verify
cmpunlocker status
# Expected output: "Memory: 40000 MB (stacks: 5)"
```

**Register Unlock as Systemd Service (Persist Across Reboots):**
```bash
# Start daemon
cmpunlocker daemon start

# Verify daemon is running
systemctl status cmpunlocker
```

**Memory Configuration Reference:**
| Configuration | CFG1 Address | LMR Address | Value |
|---------------|--------------|------------|-------|
| 10GB (stock) | 0x009A0204 | 0x00100CE0 | Default |
| 40GB (2-stack) | 0x009A0204 | 0x00100CE0 | 0x2240000 |
| 80GB (5-stack) | 0x009A0204 | 0x00100CE0 | 0x5640000 |

See cmpunlocker2 README for per-model register values.

---

## Build & Benchmark Environment

### CUDA Setup

**System Requirements:**
- CUDA 11.8+ (tested with 11.8, 12.0)
- cuDNN 8.6+
- CMake 3.24+
- C++17 compiler (GCC 10+, Clang 12+)
- NVIDIA driver 525+

**Build:**
```bash
# Create build directory
mkdir -p build && cd build

# Configure for 2080Ti (SM75)
cmake .. \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . -j$(nproc)
```

### Enable Benchmarking Mode

```bash
# Export for this session
export CUDA_VISIBLE_DEVICES=0
export TZ=UTC  # For consistent timestamping

# Optionally set GPU clock to fixed speed for reproducible benchmarks
sudo nvidia-smi -lgc 1410  # Fixed GPU clock (2080Ti max boost ~1410 MHz)
sudo nvidia-smi -pm 1     # Enable persistence mode
```

---

## Kernel Benchmarks

### 1. GQA-Aware Vector Attention (KV Reads at 262K Context)

**Benchmark Setup:**
```bash
# Single-request prefill on full context
./build/bin/benchmark-gqa-attention \
  --tokens=262144 \
  --heads=32 \
  --width=128 \
  --batch=1 \
  --iterations=5 \
  --warmup=2

# Profile with nsys for detailed analysis
nsys profile --stats=true -o gqa_262k_profile \
  ./build/bin/benchmark-gqa-attention \
    --tokens=262144 --heads=32 --batch=1 --profile=true
```

**Expected Results:**
| Metric | Measured | Physics Floor | Headroom |
|--------|----------|---------------|---------| 
| Kernel Time | ~51ms | ~18.5ms | 2.75x |
| Throughput | ~5.1M tok/s | ~14M tok/s | 2.75x |
| Memory BW | ~892 GB/s | ~1493 GB/s peak | 67% utilization |

**Analysis:**
- Bottleneck: KV cache serialization (vec-load + softmax-reduce pipeline not overlapped)
- Gap: 32.5ms (51ms - 18.5ms) of recoverable latency
- Root cause: Per-warp softmax reduction stalls lookahead KV prefetch

**Prove-It Commands:**
```bash
# Verify KV load is the bottleneck
nsys stats --report gputrace,cudaapisum gqa_262k_profile.nsys-rep \
  | grep -E "memcpy|ker_" | head -20

# Compare KV read patterns
./build/bin/test-gqa-attention --trace-kv-loads=true --tokens=262144
```

### 2. MMQ Small-Batch Performance (Batch 1-4)

**Benchmark:**
```bash
# Sweep batch sizes
for batch in 1 2 4 8; do
  echo "Batch $batch:"
  ./build/bin/benchmark-mmq-attention \
    --batch=$batch \
    --width=64 \
    --height=32 \
    --iterations=10 \
    --profile=true
done
```

**Expected Results:**
| Batch | Time (ms) | Throughput | Per-Tile Overhead |
|-------|-----------|------------|--------------------|
| 1 | ~55 | 19M elem/s | 11ms fixed |
| 2 | ~58 | 36M elem/s | 11ms fixed |
| 4 | ~64 | 64M elem/s | 11ms fixed |

**Analysis:**
- Launch overhead dominates: 11ms per GDN block launch
- Measured batch-1 floor: 55ms (no parallelism)
- Physics floor (no launches): ~38ms
- Optimization target: Fused tile-loop + stream pipelining

### 3. GDN Hybrid Block (Launch Overhead & State Indexing)

**Benchmark:**
```bash
# Measure launch overhead per block
./build/bin/benchmark-gdn-hybrid \
  --hybrid-blocks=1 \
  --measure-launch=true \
  --iterations=100 \
  --report-timeline=true

# Analyze launch delays
python scripts/analyze-launch-timeline.py
```

**Expected:**
- Per-block launch: ~11ms
- State indexing (ragged seqlens): ~2ms per request  
- Stride computation: ~1ms

**Optimization:**
- Replace per-block launch with global kernel-per-request
- Cache stride computation across requests

---

## Multi-GPU Scaling (Tensor Parallelism)

### Dual 2080Ti Setup

**Hardware Configuration:**
```bash
# Verify both GPUs visible
nvidia-smi -L

# Check PCIe topology
nvidia-smi topo --matrix
```

**Expected topology:**
```
        GPU0 GPU1
GPU0     X   PXB
GPU1    PXB   X
```
PXB = PCIe bridge (16 GB/s nominal, ~12 GB/s effective)

**Build with TP Support:**
```bash
cmake .. \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DENABLE_TENSOR_PARALLEL=ON \
  -DPARTITION_SIZE=2
```

**Benchmark Multi-GPU:**
```bash
# TP=2 inference on Qwen3.6-27B-AWQ (262K context)
export CUDA_VISIBLE_DEVICES=0,1
./build/bin/benchmark-tp \
  --model=qwen-27b-awq \
  --context-length=262144 \
  --batch=1 \
  --partition-size=2 \
  --iterations=10

# Expected throughput:
# - Prefill: ~1841 tok/s (dual GPU)
# - Decode: ~101 tok/s (dual GPU)
```

**Performance Analysis:**
| Phase | Single GPU | Dual GPU (TP) | Scaling |
|-------|-----------|---------------|---------|
| Prefill | 1200 tok/s | 1841 tok/s | 1.53x |
| Decode | 60 tok/s | 101 tok/s | 1.68x |
| All-reduce | — | 3-5 GB/s | PCIe limited |

Scaling < 2x due to all-reduce overhead and PCIe saturation.

---

## Multi-Token Prediction (Speculative Decoding)

Tenselerate includes MTP kernel integration for speculative decoding.

**Benchmark:**
```bash
# LongGen3 MTP sweep (K=1..5)
./build/bin/benchmark-mtp \
  --model=qwen-27b \
  --context-length=262144 \
  --k-values=1,2,3,4,5 \
  --iterations=20

# Expected acceptance rates:
# K=1: 100% (baseline)
# K=2: 87% average
# K=3: 74% average (conservative deploy point)
# K=4: 62% average
# K=5: 51% average (falling off)
```

**Deployment Recommendation:**
Use **K=3** for production. Balances:
- 2.1x throughput improvement
- 74% token acceptance (low re-run cost)
- Stable across context lengths

---

## Performance Validation

### Pre-Benchmark Checklist

```bash
#!/bin/bash

# 1. Verify GPU memory
nvidia-smi -q -d MEMORY | grep "Used\|Free"
# Expected for unlocked 2080Ti: Total 40960 MB, Free ~38000 MB

# 2. Check clocks are stable
nvidia-smi --query-gpu=clocks.current.graphics,clocks.current.memory --format=csv
# Should show ~1410 MHz GPU, ~7000 MHz memory

# 3. Verify no other processes
ps aux | grep cuda
# Should be empty

# 4. Check thermal headroom
nvidia-smi --query-gpu=temperature.gpu --format=csv
# Should be < 50C before benchmarking

# 5. Disable dynamic clocking for reproducibility
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 1410
```

### Benchmark Template

```bash
for run in {1..3}; do
  echo "Run $run:"
  nsys profile \
    --sample=none \
    --trace=cuda,osrt \
    --duration=10s \
    -o profile_run_${run}.nsys-rep \
    ./build/bin/benchmark-kernel --profile=true
done
```

---

## Profiling & Analysis

### Using NSys

```bash
# Collect profile
nsys profile --stats=true -o kernel_profile \
  ./build/bin/benchmark-kernel

# Analyze memory transfer sizes
nsys stats --report gputrace kernel_profile.nsys-rep | grep -E "memcpy|kernel" | head -30

# Timeline view (export to SpeedScope)
nsys export --output=json -o kernel_profile.json kernel_profile.nsys-rep
```

### Using Nsight Compute

```bash
# Profile single kernel
ncu -k regex:kernel_name \
    -o kernel_detail.ncu-rep \
    ./build/bin/benchmark-kernel

# Import in Nsight Compute GUI for detailed metrics
```

---

## Common Issues & Solutions

### Memory Unlock Reverted After Reboot

**Solution:** Ensure cmpunlocker daemon is enabled:
```bash
sudo systemctl enable cmpunlocker
sudo systemctl start cmpunlocker
sudo systemctl status cmpunlocker
```

### Kernel Crashes with Unlocked Memory

**Cause:** Register offset mismatch for your specific 2080Ti SKU (different PCB revisions)
**Solution:** 
```bash
# Check your exact register values
cmpunlocker status --verbose

# If mismatch, report to cmpunlocker2 project with full details
```

### PCIe Bandwidth Saturated in TP=2

**Expected:** 12 GB/s effective is the PCIe 3.0 x16 limit
**Mitigation:**
- Use all-reduce fusion (gradient accumulation pipeline)
- Reduce precision (FP16 all-reduce instead of FP32)
- Consider single GPU with longer context instead

### Benchmarks Don't Match Expected Physics Floor

**Checklist:**
- [ ] cmpunlocker still active: `cmpunlocker status`
- [ ] Clock fixed: `nvidia-smi --query-gpu=clocks.current.graphics`
- [ ] No background processes: `nvidia-smi pmon`
- [ ] No throttling: check POWER in `nvidia-smi`
- [ ] Kernel is running: `nsys profile` to verify kernel launches

---

## Next Steps

See the Tenselerate kernel-work roadmap in `docs/kernel-work.md` for open optimization items and reproducible prove-it commands for each bottleneck.

Also consult `docs/physics.md` for detailed speed-of-light analysis, HBM2e bandwidth modeling, and achievable headroom per subsystem.

For full build & architecture docs, see `docs/build.md` and `HERCULES.md`.
