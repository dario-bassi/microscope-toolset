### Roadmap
---

# (2026-04-30)

---

## Workstream A — Proxy / Core Separation

Goal: move the simulation (and eventually the real microscope core) behind a network
boundary so the Claude agent cannot read simulation source files, test configs, or
directly import virtual device classes.

### A1. cfg-file classifier
Write `_classify_cfg(path) -> Literal["real", "virtual", "mixed"]` in `mcp_server_gui.py`.

- No `#py` lines → `"real"` (pure C++ drivers)
- All device lines have `#py` → `"virtual"` (pure Python devices)
- Mix of `#py` and non-`#py` → `"mixed"` (block with error, not supported yet)

Replaces `_is_old_style_virtual_cfg()` and any ad-hoc detection logic.

### A2. SimServerWorker — blocked by A1
Add `SimServerWorker(QObject)` to `mcp_server_gui.py`.

- Constructor: `backend: str`, `params: dict`, `port: int = 5601`
- `run()` slot (executes in its own `QThread`):
  1. Lazy-imports `virtual_microscope` and `pymmcore_proxy`
  2. Calls `load_backend(backend, **params)` — registers Python devices in-process on the server
  3. Starts `serve(core, port=port)` in a daemon thread
  4. Polls `GET /health` until ready (max ~10 s)
  5. Emits `server_ready(url: str)` or `server_error(msg: str)`

The Qt main thread never blocks.

### A3. Refactor `_capturing_load()` — blocked by A1, A2

| cfg type | Action |
|----------|--------|
| `"real"` | `_nm_load(path)` as today → `_start_phase2()` |
| `"virtual"` | Start `SimServerWorker` in `QThread`, return early; Phase 2 triggered from slot |
| `"mixed"` | Emit error status, return without loading |

Remove the old `_is_old_style_virtual_cfg` branch once confirmed dead.

### A4. Wire `_on_sim_server_ready` slot — blocked by A3

```python
@pyqtSlot(str)
def _on_sim_server_ready(self, url: str):
    remote_core = connect(url)       # RemoteMMCore
    win.set_core(remote_core)        # napari-mm now controls remote core
    self._in_user_load = False
    self._start_phase2()             # MCPWorker gets RemoteMMCore

@pyqtSlot(str)
def _on_sim_server_error(self, msg: str):
    self._in_user_load = False
    self.status_update.emit(f"Simulation server error: {msg}")
```

Verify `_patched_set_core` re-installs `_capturing_load` hooks on `RemoteMMCore`.

### A5. Remove `src/virtual_microscope/` — blocked by A4
1. Confirm no cfg files reference `src.virtual_microscope` device lines
2. Check benchmarking tests for direct imports
3. Delete `src/virtual_microscope/` directory
4. Remove `_extract_cell_type()` helper
5. Update `configs/virtual_*.cfg` as needed

### A6. Test isolation server — blocked by A4
Create `src/benchmarking/test_server.py` exposing a small HTTP API:
- `GET /test/{id}/config` — returns test metadata
- `POST /test/{id}/initialize` — loads backend + sets initial device properties
- `GET /test/{id}/status` — current test state

`test_runner.py` calls this server instead of importing `initialize_test.py` directly.
The MCP agent has no route to test config files.

---

## Workstream B — Database / Storage Refactor

Goal: make Elasticsearch and PostgreSQL fully optional, remove the OpenAI API
dependency from the live agent path, and give the user a proper UI for managing
knowledge bases.

### B1. Remove OpenAI API dependency
- Remove `openai.OpenAI` client from `agents_init.py` and `specialized_agent.py`
- Remove embedding generation (`_embeds_query`) — fall back to BM25-only search
- Remove `base_agent.py` OpenAI structured output calls
- Make `OPENAI_API_KEY` optional in `.env` and `get_user_information()`

### B2. Make Elasticsearch and PostgreSQL optional — blocked by B1
- Add settings flags: `use_elasticsearch`, `use_postgresql` (in `.env` or settings UI)
- `agents_init.py`: skip `DatabaseAgent` init if ES disabled; MCP knowledge tools
  return graceful "database not configured" message
- `MCPWorker`: skip ES startup if disabled
- `src/postqrl/`: only import/init if `use_postgresql=true`
- What each backend stores:
  - **Elasticsearch** → API docs, MM device docs, PDFs, user-uploaded documents
  - **PostgreSQL** → successful agent workflows + generated code

### B3. Database management UI panel — blocked by B2
Separate `QWidget` for one-time database setup:
- ES section: index status, doc count, "Rebuild index" button
- PG section: connection status, workflow count, "Clear collection" button
- Upload section → triggers ingestion pipeline (see B4)
- All database indexing lives here; `agents_init.py` only reads, never writes

### B4. Document ingestion pipeline (txt, docx, pdf) — blocked by B2
Create `src/databases/ingestion.py`:

```
load_document(path)
    → .txt  : direct read
    → .docx : python-docx
    → .pdf  : pdfplumber / pymupdf

extract_text_and_images(doc)
    → [{type: "text"|"image", content, page, metadata}]

chunk_text(text, chunk_size=512, overlap=64)
    → overlapping chunks preserving section/page metadata

index_document(path, index_name, es_client)
    → full pipeline: load → extract → chunk → (embed) → bulk index
```

Chunk metadata: `source_file`, `page_number`, `chunk_index`, `document_type`, `creation_date`.

### B5. Elasticsearch document browser UI — blocked by B3, B4
`QWidget` inside the database management panel:
- Document list grouped by type (pymmcore API, MM devices, PDFs, user docs)
- Document detail: chunks, page numbers, extracted images
- Similarity connections: "related docs" via ES more-like-this or stored KNN neighbours
- Upload → ingestion pipeline with progress indicator
- Delete → removes document and all chunks from index
- Index stats: total docs, total chunks, last updated

### B6. PostgreSQL UI decision
Options:
- **Option A** (start here): point users to pgAdmin / DBeaver — zero build cost, full
  SQL power. Document schema clearly so any SQL tool works.
- **Option B** (future): minimal read-only in-app viewer: list workflows, show code +
  result, search by embedding similarity.

Decide based on how frequently workflows need to be browsed operationally.

---

## Workstream C — Code Execution Safety

Goal: fail fast with clear messages, prevent the agent from bypassing hardware safety
boundaries, add library-specific performance guards.

### C1. Pre-execution import validation
Replace `_preimport_dependencies()` silent-install with fail-fast AST check:
- `importlib.util.find_spec()` for each imported module
- If missing, return early:
  `"Missing packages: [x, y, z]. Ask the user whether they should be installed."`
- No auto-install, no subprocess pip call

### C2. User-confirmation package install — blocked by C1
Add `request_package_install(packages: list[str])` MCP tool:
- Qt dialog on main thread: `"Agent wants to install: x, y. Allow?"`
- If approved: `pip install`, return success/failure
- If denied: return `"User declined installation of [x, y]"` so agent adapts

### C3. Block unsafe patterns in agent code
Extend static analysis in `Execute` alongside `is_safe_viewer()`:

| Blocked pattern | Reason |
|----------------|--------|
| `CMMCorePlus.instance()`, `UniMMCore()`, `RemoteMMCore()` | Agent must use pre-bound `mmc` |
| `mmc.loadSystemConfiguration(...)` | cfg selection is a user decision |
| `import virtual_microscope` | simulation internals not agent-accessible |
| `import pymmcore_proxy` | proxy internals not agent-accessible |

Each block returns a descriptive message so the agent self-corrects.
Keep `is_safe_viewer()` unchanged.

### C4. Library-specific execution guards — blocked by C3
Pluggable registry: `LIBRARY_GUARDS = {"cellpose": check_cellpose_args, ...}`

**Cellpose guard**: if `model.eval()` / `model.run()` called with image > configurable
threshold (default 2048×2048 px):
- Warn and recommend downscaling first
- Escalate to full resolution only if user confirms or initial result is insufficient
- Hard block only for images > 8192×8192 (would clearly hang)

Document all active guards in `CLAUDE.md`.

---

## Workstream D — Microscope Metadata & Agent Feedback

Goal: make hardware events visible to the agent automatically during code execution
without requiring explicit `get_microscope_events()` calls.

### D1. Design: proactive event push

| Option | Mechanism | Effort |
|--------|-----------|--------|
| A — implement first | Append event log to `execute_python_code` return value | Low |
| B — long-term goal | MCP SSE/WebSocket push notifications to Claude client | High |
| C — alternative | Pre/post `get_microscope_settings()` diff in tool result | Medium |

### D2. Append event log to execute result — blocked by D1
After every `execute_python_code` call:
1. Record `start_time` before execution
2. Drain `MicroscopeEventCache` for events after `start_time`
3. If events exist, append to return string:

```
--- Microscope events during execution ---
14:23:01.452  exposureChanged        Camera    50ms → 100ms
14:23:01.891  XYStagePositionChanged XYStage   x=120.0 y=85.0
```

4. Wire same drain into `snap_image()` result for shutter/exposure events.
5. If no events: omit section entirely (no noise).

---

## Workstream E — Experiment Workspace

Goal: every agent session gets a known output folder, pre-bound in the execution
namespace, so the agent never needs to hardcode or guess paths.

### E1. Session workspace folder — blocked by A4
When Phase 2 starts (`_on_config_loaded`):
- Auto-create: `experiments/YYYYMMDD_HHMMSS_{backend_name}/`
- Pre-bind in `Execute` namespace: `self.namespace["experiment_dir"] = workspace_path`
- Add MCP tool `get_experiment_workspace()` → returns current session folder path
- Log workspace path in status label and `microscope_toolset.log`

Pre-flight scan: detect hardcoded paths in `open()`, `tifffile.imwrite()`, `np.save()`
calls that don't exist → warn agent to use `experiment_dir` instead.

---

## Workstream F — Audits (independent, no blockers)

### F1. Audit `src/postqrl/`
- Find all live call sites of `DBConnection` and `LoggerDB`
- If unused: remove directory, clean up `agents_init.py` imports
- If used: document schema; check whether it duplicates Elasticsearch

### F2. Audit `src/local/prepare_code.py`
- Find all call sites of `prepare_code()`, `contains_instance_object()`, etc.
- If no longer effective: delete module and remove import in `server_setup.py`

### F3. Audit `src/agentsNormal/` OpenAI usage
- Confirm whether `OPENAI_API_KEY` is required at runtime
- Check if `base_agent.py:call_agent()` is in any live MCP tool path
- Document decision: keep OpenAI for embeddings + cross-encoder reranking, or migrate

---

## Dependency Graph

```
A1 → A2 → A3 → A4 → A5
                A4 → A6
                A4 → E1

B1 → B2 → B3 → B5
      B2 → B4 → B5
      B6 (independent decision)

C1 → C2
C3 → C4

D1 → D2

F1, F2, F3  (no blockers — do anytime)
```

---

## Open Questions

- **Embedding strategy after OpenAI removal**: BM25-only is a safe fallback. Longer
  term: evaluate `sentence-transformers` locally or Claude's embedding API.
- **Mixed cfg files**: blocked for now. Future option: sim server handles `#py` devices
  while `CMMCorePlus` handles C++ drivers, requiring a multiplexing proxy layer.
- **MCP push notifications (Option B)**: requires FastMCP + Claude client to support
  SSE or WebSocket event streams. Track pymmcore-proxy and MCP spec updates.
- **PostgreSQL schema**: document column names and vector dimensions in
  `docs/postgresql_schema.md` so pgAdmin/DBeaver users can query without reading code.
- Created different tool to use in the GUI