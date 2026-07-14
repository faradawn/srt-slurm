#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# AIPerf concurrency-sweep benchmark.
#
# Drives the serving frontend with NVIDIA aiperf (high-performance client) instead of
# sa-bench's single-process Python client, which under-reports high-concurrency throughput
# due to SSE streaming backpressure. Mirrors the InferenceX reference aiperf command.
#
# Args (positional, from AIPerfSweepRunner.build_command):
#   1 ENDPOINT           http://localhost:<frontend_port>
#   2 MODEL_NAME         served model name
#   3 TOKENIZER_PATH     HF id or /model
#   4 ISL                input seq len
#   5 OSL                output seq len
#   6 CONCURRENCIES      x-separated list, e.g. "1x8x64x512"
#   7 NUM_PROMPTS_MULT   request-count = concurrency * this
#   8 NUM_WARMUP_MULT    warmup-request-count = concurrency * this
#   9 ENDPOINT_TYPE      aiperf endpoint type (default: chat)
set -uo pipefail

ENDPOINT="${1:?endpoint}"
MODEL_NAME="${2:?model}"
TOKENIZER_PATH="${3:?tokenizer}"
ISL="${4:?isl}"
OSL="${5:?osl}"
CONCURRENCIES="${6:?concurrencies}"
NUM_PROMPTS_MULT="${7:-3}"
NUM_WARMUP_MULT="${8:-1}"
ENDPOINT_TYPE="${9:-chat}"

ARTIFACT_BASE="/logs/artifacts"
mkdir -p "$ARTIFACT_BASE"

echo "=============================================="
echo "AIPerf concurrency-sweep benchmark"
echo "  Endpoint:      $ENDPOINT ($ENDPOINT_TYPE)"
echo "  Model:         $MODEL_NAME"
echo "  ISL/OSL:       $ISL / $OSL"
echo "  Concurrencies: $CONCURRENCIES"
echo "=============================================="

# Install aiperf into an isolated venv that inherits system site-packages, so we do NOT
# uninstall distutils-installed packages (e.g. blinker 1.4) inside the serving container,
# which makes a bare `pip install aiperf` fail.
AIPERF_VENV="/tmp/aiperf_venv"
if [ ! -x "$AIPERF_VENV/bin/aiperf" ]; then
    echo "Installing aiperf into $AIPERF_VENV ..."
    python3 -m venv --system-site-packages "$AIPERF_VENV"
    "$AIPERF_VENV/bin/pip" install --no-cache-dir "${AIPERF_PACKAGE:-aiperf}" 2>&1 | tail -3
fi
AIPERF="$AIPERF_VENV/bin/aiperf"
"$AIPERF" --version || { echo "aiperf install failed"; exit 1; }

IFS='x' read -ra CONC_LIST <<< "$CONCURRENCIES"
FAIL=0
for conc in "${CONC_LIST[@]}"; do
    reqcount=$(( conc * NUM_PROMPTS_MULT ))
    warmup=$(( conc * NUM_WARMUP_MULT ))
    outdir="$ARTIFACT_BASE/conc_${conc}"
    mkdir -p "$outdir"
    echo ""
    echo "$(date '+%Y-%m-%d %H:%M:%S') - aiperf concurrency=$conc requests=$reqcount warmup=$warmup"
    set -x
    "$AIPERF" profile \
        -m "$MODEL_NAME" \
        --tokenizer "$TOKENIZER_PATH" \
        --url "$ENDPOINT" \
        --endpoint-type "$ENDPOINT_TYPE" \
        --ui-type none \
        --streaming \
        --concurrency "$conc" \
        --request-count "$reqcount" \
        --warmup-request-count "$warmup" \
        --isl "$ISL" --osl "$OSL" \
        --use-server-token-count \
        --extra-inputs ignore_eos:true \
        --artifact-dir "$outdir" || FAIL=1
    set +x
done

echo ""
echo "AIPerf sweep complete (fail=$FAIL). Artifacts under $ARTIFACT_BASE"
exit $FAIL
