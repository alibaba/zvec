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
