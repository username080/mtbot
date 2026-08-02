"""
"Unused code" regression tests.

While reviewing the Manager/DataPipeline/Analyser/Executer integration we
found several functions and constants that are defined but never used
anywhere else in the codebase - either because a feature was wired up
incompletely (check_time_stops), a routing helper nothing ever calls
(RedisManager.distribute), or a vestigial constant left behind by a
refactor (SHM_SIZE).

Some previously-tracked entries here have since been resolved by deleting
the dead code outright rather than leaving it unreferenced: read_macro_bars
/read_micro_bars/get_stats on DataPipelineService; create_packet (whole
file manager/utils/manager_utils.py deleted); and AnalyserIPCManager.
send_to_executer (whole file workers/analyser/ipc_manager.py deleted, once
ExecutorService grew its own persistent connection and made the separate
"IPC manager" class redundant). That's the tripwire doing its job, per
point 2 below.

Rather than leaving that as a one-off note, this file turns each finding
into a small automated tripwire:

  1. If someone starts calling one of these again without updating this
     file, nothing breaks here - that's fine, it just means the code is no
     longer dead.
  2. If someone deletes the dead code, or properly wires the feature up,
     THIS TEST WILL START FAILING. That's the point - it's a deliberate
     tripwire. When it fails, the fix is to remove that entry from
     DEAD_CODE_INVENTORY below (not to "fix the test"), because the entry
     has done its job and the finding is resolved.

We check "is this referenced anywhere outside its own definition file" with
a plain whole-word text search across the repo's .py files (skipping venv/,
__pycache__/, and this tests/ directory). That's a blunt instrument - no
real call-graph or AST analysis - but that's the right level of rigor here:
we only need to know "does *any* other line of code even mention this name."
"""
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Each entry: (file the symbol is defined in, relative to repo root; symbol name)
DEAD_CODE_INVENTORY = [
    # Full "Microbar Supervisor" time-stop safety net - implemented, but
    # nothing in Analyser.work()'s main loop ever calls it.
    ("workers/analyser/strategy_coordinator.py", "check_time_stops"),

    # RedisManager exposes send/receive/listen (all wired up) plus this
    # routing helper, which nothing calls.
    ("manager/redis_manager.py", "distribute"),

    # Vestigial constant from before shm_service moved to per-segment
    # sizes (portfolio_size/positions_info_size).
    ("workers/base_worker.py", "SHM_SIZE"),
]

_SKIP_DIRS = {"venv", "__pycache__", ".git", "tests"}


def _iter_repo_python_files():
    for path in REPO_ROOT.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in path.relative_to(REPO_ROOT).parts):
            continue
        yield path


def _count_references(symbol: str, defining_file: pathlib.Path) -> int:
    """Count files (other than `defining_file`) that mention `symbol` as a whole word."""
    pattern = re.compile(rf"\b{re.escape(symbol)}\b")
    hits = 0
    for path in _iter_repo_python_files():
        if path == defining_file:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if pattern.search(text):
            hits += 1
    return hits


@pytest.mark.parametrize("defining_file, symbol", DEAD_CODE_INVENTORY)
def test_symbol_is_still_unused_elsewhere(defining_file, symbol):
    """
    Documents a currently-dead symbol.

    If this fails, `symbol` is now referenced somewhere else in the codebase
    - go remove its entry from DEAD_CODE_INVENTORY above. That's a good
    thing, not a bug in this test.
    """
    full_path = REPO_ROOT / defining_file
    assert full_path.exists(), (
        f"{defining_file} no longer exists - '{symbol}' was presumably deleted "
        f"along with it. Remove this entry from DEAD_CODE_INVENTORY."
    )

    reference_count = _count_references(symbol, full_path)
    assert reference_count == 0, (
        f"'{symbol}' (defined in {defining_file}) is now referenced in "
        f"{reference_count} other file(s) - looks like it's wired in now. "
        f"Remove it from DEAD_CODE_INVENTORY in this test file."
    )


def test_shm_name_and_lock_constructor_args_are_dead(isolated_shm):
    """
    Manager passes shm_name=/shm_lock= into every worker's constructor (see
    manager/manager.py: DataPipeline(..., shm_name=self.shm_name,
    shm_lock=self.shm_lock)), and BaseWorker.__init__ stores them onto
    shm_service.shm_name / shm_service.lock.

    But SharedMemoryService has since moved to separately-named segments
    (portfolio_name / positions_info_name), each read/written through its
    own dedicated lock -
    read_portfolio()/write_portfolio()/etc. never look at
    shm_service.shm_name or shm_service.lock. So today, the shm_name/
    shm_lock arguments threaded through every worker's constructor do
    nothing at all.

    Proven behaviorally here (not by reading source text): overwriting
    those two attributes with garbage has zero effect on where the
    portfolio segment is actually read from.
    """
    original_portfolio_name = isolated_shm.portfolio_name

    # This is exactly what BaseWorker.__init__ does with the shm_name/
    # shm_lock constructor arguments Manager passes to every worker:
    isolated_shm.shm_name = "this-value-should-be-irrelevant"
    isolated_shm.lock = None  # even a nonsensical, unusable "lock"

    # If shm_name/lock actually controlled portfolio reads, this would now
    # raise (None has no __enter__) or read the wrong segment. It doesn't -
    # proving those two attributes are dead weight.
    portfolio = isolated_shm.read_portfolio()
    assert portfolio == {"cash": 100_000.0, "positions": {}}
    assert isolated_shm.portfolio_name == original_portfolio_name
