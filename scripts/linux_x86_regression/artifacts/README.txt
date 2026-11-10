Current final handoff (2026-11-09)
=================================
Production commit 008fbdc36737e0b4b70ee2e29a3ff12889f8061a now includes the
Linux final INT4 kernels and both macOS ARM64 corrections (large query tiles
and FP16 1x1 IP direct calls). Start with ../FINAL_X86_AGENT_START_HERE_zh.txt.
Reports and the historical/final status map: ../reports/README_zh.md.

New checked-in evidence archives (each has a .sha256 sidecar):
  macos-residual-final-evidence-20261109.tar.gz
  macos-arm64-guard-evidence-20261109.tar.gz
  macos-fp16-pair-evidence-20261109.tar.gz
Each contains raw measurements, correctness XML/logs, 200 parent fixtures,
disassembly, build information and EVIDENCE_SHA256SUMS. The final package also
contains the dispatch probe and unchanged x86 cross-compiled objects.

The original packages and sections below are historical snapshots. Statements
such as "NOT applied" or "pending macOS" refer to their preparation time.
Do not apply their candidate patches on top of the current branch. New server
baseline patches are under ../final_revalidation/ and target detached 955 trees.
Linux server revalidation of the committed final tree (branch a934deea, code
008fbdc3) was completed on 2026-11-09/10:
  linux-x86-final-revalidation-results.tar.gz (+ .sha256, inner evidence/SHA256SUMS)
Report: ../reports/linux_final_revalidation_20261110/REPORT_zh.md. macOS-side
verification instructions: ../MACOS_AGENT_VERIFY_LINUX_FINAL_zh.txt.

Historical handoffs
===================

x86 follow-up handoff, prepared 2026-11-06

x86-followup-handoff.tar.gz contains the candidate patch against 955ae841,
the baseline test build patch, regression toolkit, execution instructions,
macOS evidence and a SHA256SUMS manifest. The package records the status at
preparation time, before it and the candidate were committed to this branch.

After pulling fix/legacy-matrix-distance-performance on the server:

  cd scripts/linux_x86_regression/artifacts
  sha256sum -c x86-followup-handoff.tar.gz.sha256
  tar -xzf x86-followup-handoff.tar.gz
  cd x86-followup
  sha256sum -c SHA256SUMS

Follow toolkit/X86_FOLLOWUP_AGENT_INSTRUCTIONS_zh.txt. Set PKG to this extracted
directory. Apply the bundled patch to the specified detached baseline, not
to the branch checkout which already contains the candidate changes.

Validation status: macOS correctness passed; x86 execution is pending.
One macOS INT16 IP microbenchmark still needs investigation. See
MACOS_STATUS.txt for scope and results; this is not a two-platform pass.

Residual follow-up, prepared 2026-11-09
------------------------------------
residual-performance-handoff.tar.gz contains the next incremental candidate,
baseline patches, toolkit and macOS evidence. Start with
../RESIDUAL_START_HERE_zh.txt, then follow the detailed instructions in the
extracted package. Verify residual-performance-handoff.tar.gz.sha256 first.
This archive was frozen before its commit; its preparation-time status text
does not mean that the checked-in candidate still needs to be committed.
Linux GCC/x86 validation remains pending.

Residual final candidate (x86 validated), prepared 2026-11-09
-------------------------------------------------------------
x86-residual-final-handoff.tar.gz contains the final Linux x86 candidate
(patches/final_vs_9819c706.patch, matrix.h only), the x86 report, two-round
comparison tables, correctness summaries, disassembly evidence and the
diagnostic variants. Start with ../MACOS_RESIDUAL_FINAL_START_HERE_zh.txt.
Verify x86-residual-final-handoff.tar.gz.sha256 first. The candidate is NOT
applied to the branch; macOS/ARM64 must re-validate it (matrix.h is shared
code) before a two-platform pass can be claimed.
