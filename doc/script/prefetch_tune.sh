#!/bin/bash
# =============================================================================
# prefetch_offset / prefetch_lines 甜点值搜索脚本
#
# 用法: bash doc/prefetch_tune.sh
#
# 原理: 修改 VectorDBBench zvec.py 中的 extra_params 并运行 benchmark，
#       收集每组参数的 QPS 结果，最后按 QPS 排序输出甜点值。
# =============================================================================

set -euo pipefail

# --------------- 配置 ---------------
ZVEC_PY="/root/code/VectorDBBench/vectordb_bench/backend/clients/zvec/zvec.py"
RESULT_DIR="/root/code/zvec/doc/prefetch_results"
LOG_DIR="/root/code/zvec/doc/prefetch_logs"

# 基础命令 (来自 draft.md.bak L334-344)
BASE_CMD="vectordbbench zvec --path /root/code/VectorDBBench/db/cohere-1m-exp --db-label 16c64g-v0.1 --case-type Performance768D1M --num-concurrency 16 --quantize-type int8 --m 15 --ef-search 135 --skip-drop-old --skip-load"

# 要扫描的参数网格
# prefetch_offset: 预取邻居向量数 (main 默认 24)
# prefetch_lines:  每个向量预取 cacheline 数 (0 = auto)
OFFSETS=(16 24 32)
PL_VALUES=(0 1 2 3)

# --------------- 工具函数 ---------------
mkdir -p "$RESULT_DIR" "$LOG_DIR"

patch_zvec_py() {
    local po="$1" pl="$2"
    # 用 python 精确替换 zvec.py 中 extra_params 块
    python3 -c "
import re, sys
with open('$ZVEC_PY', 'r') as f:
    content = f.read()
pattern = r'\"prefetch_offset\":\s*\d+,\s*\n\s*\"prefetch_lines\":\s*\d+,'
replacement = '\"prefetch_offset\": $po,\n                    \"prefetch_lines\": $pl,'
content = re.sub(pattern, replacement, content)
with open('$ZVEC_PY', 'w') as f:
    f.write(content)
"
}

extract_qps() {
    # 从日志中提取 QPS (匹配 "qps=13775.6395" 格式)
    local log_file="$1"
    local qps
    qps=$(grep -oP 'qps[=:]\s*\K[\d.]+' "$log_file" 2>/dev/null | tail -1)
    echo "${qps:-N/A}"
}

# --------------- 主流程 ---------------
echo "=============================================="
echo " Prefetch Tuning Sweep"
echo " offsets: ${OFFSETS[*]}"
echo " lines:   ${PL_VALUES[*]}"
echo "=============================================="
echo ""

# 保存原始 zvec.py
cp "$ZVEC_PY" "${ZVEC_PY}.bak"

TOTAL=$((${#OFFSETS[@]} * ${#PL_VALUES[@]}))
IDX=0

# 写入 CSV 头
echo "prefetch_offset,prefetch_lines,qps,elapsed_sec" > "$RESULT_DIR/results.csv"

for po in "${OFFSETS[@]}"; do
    for pl in "${PL_VALUES[@]}"; do
        IDX=$((IDX + 1))
        TAG="po${po}_pl${pl}"
        LOG_FILE="$LOG_DIR/${TAG}.log"

        echo "[$IDX/$TOTAL] prefetch_offset=$po, prefetch_lines=$pl"
        echo "  Patching zvec.py..."
        patch_zvec_py "$po" "$pl"

        # 验证 patch
        if ! grep -q "\"prefetch_offset\": $po" "$ZVEC_PY"; then
            echo "  ERROR: patch failed, skipping"
            echo "$po,$pl,PATCH_FAIL,0" >> "$RESULT_DIR/results.csv"
            continue
        fi

        echo "  Running benchmark..."
        START_TS=$(date +%s)

        # 运行 benchmark (stdout + stderr 都记录)
        if eval "$BASE_CMD" > "$LOG_FILE" 2>&1; then
            END_TS=$(date +%s)
            ELAPSED=$((END_TS - START_TS))
            QPS=$(extract_qps "$LOG_FILE")
            echo "  -> QPS=$QPS  (${ELAPSED}s)"
            echo "$po,$pl,$QPS,$ELAPSED" >> "$RESULT_DIR/results.csv"
        else
            END_TS=$(date +%s)
            ELAPSED=$((END_TS - START_TS))
            echo "  -> FAILED (${ELAPSED}s)"
            echo "$po,$pl,FAIL,$ELAPSED" >> "$RESULT_DIR/results.csv"
        fi

        echo ""
    done
done

# 恢复原始 zvec.py
cp "${ZVEC_PY}.bak" "$ZVEC_PY"
rm -f "${ZVEC_PY}.bak"

# --------------- 汇总 ---------------
echo "=============================================="
echo " RESULTS SUMMARY"
echo "=============================================="
echo ""
echo "All results:"
column -t -s',' "$RESULT_DIR/results.csv"
echo ""
echo "Top 5 by QPS (descending):"
head -1 "$RESULT_DIR/results.csv"
tail -n+2 "$RESULT_DIR/results.csv" | grep -v FAIL | grep -v PATCH_FAIL | sort -t',' -k3 -rn | head -5 | column -t -s','
echo ""
echo "Results saved to: $RESULT_DIR/results.csv"
echo "Logs saved to:    $LOG_DIR/"
