Portable baseline patches and correctness audit for production 008fbdc3.
Start with ../FINAL_X86_AGENT_START_HERE_zh.txt.

All patches in patches/ apply to detached 955ae841 trees, not to the current
branch. versions.json records each resulting src/tests tree and patch SHA256.
The candidate source/test trees were checked against the production commit.
The Linux final patch was checked against the original verified handoff.

verify_correctness.py audits the existing run.py results. The known-parent
flag requires manual review of the historical INT8 IP failures first; it does
not allow any failed fixed test or missing fixture/XML. Parent failures remain
in correctness.json and in the separate audit report. Missing fixture reads
are executed only after all other required checks succeed.

Local validation: Python syntax/help checks; seven audit scenarios using
recorded macOS XML and synthetic failures (successful audit, failed fixed test,
missing XML, missing explicit exception flag, completing a skipped fixture read
while preserving parent failure, unexpected parent case, 199 fixtures).
The read-completion branch used a test stub returning recorded XML; this is a
validator self-check, not a Linux execution result. All scenarios passed.

Linux server execution of this new handoff remains pending.
