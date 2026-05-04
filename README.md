# Microscope Toolset
This repository is a toolset for microscope that use pymmcore-plus with LLM

---

> ⚠️ **Security & Liability Warning**
>
> This toolset gives an LLM agent direct access to your computer: it can execute arbitrary Python code, read and write files, control microscope hardware, and install packages into your Python environment.
>
> **You are responsible for reviewing every action the agent proposes before approving it.** In particular:
> - The `install_packages` tool will install packages directly into your active conda/uv environment. Only approve packages you recognise and trust.
> - The `execute_python_code` tool runs code with the same privileges as your user account.
>
> The author(s) of this project accept **no responsibility** for any damage, data loss, security incident, or unintended hardware interaction that may result from using this software. Use it at your own risk.

---

### How to get  started

In order to use this toolset, you need to have python installed. There are differents option, you could use *Anaconda*, *miniconda* or *mamba*. After you installed your favorite package and environment management system create a specific environment for this toolset.

```
conda create -n microscope-toolset python=3.12.11
```
Afterwards activate your newly environment
```
conda activate microscope-toolset
```
Create a new folder and clone the _microscope-toolset_ repository
```
git clone https://github.com/ddd42-star/microscope-toolset.git
```
Then go into the folder of this repository and install all the packages using the *requirements* file
```
pip install -r requirements.txt
```

### Installation
```
pip install -e .
```
or 
```
python -m compileall src/
```

#### Optional databases: Elasticsearch and PostgreSQL

Both Elasticsearch and PostgreSQL are **optional**. The toolset starts and runs without them — database-backed features (semantic search, session logging) are simply skipped when the services are not configured.

A full setup and usage guide for both databases is under implementation. For now, set the relevant variables in your `.env` file to enable them:

**Elasticsearch** — used for semantic search over API docs and publications:
```
ELASTICSEARCH="<path to your elasticsearch installation>"   # enables auto-start of the ES server
ELASTICSEARCH_URL="http://localhost:4500"                   # connection URL (default shown)
```

**PostgreSQL** — used for logging agent sessions:
```
DB_HOST="localhost"
DB_PORT=5432
DB_NAME="<your database name>"
DB_USER="<your user>"
DB_PASSWORD="<your password>"
```

If any of the `DB_*` variables are absent, the PostgreSQL logger is skipped automatically.

### Model configuration

The toolset uses two models, both configurable via the `.env` file:

**1. Claude (Anthropic) — for query reformulation inside the database agent**

Set your API key and choose a model:
```
ANTHROPIC_API_KEY="your-anthropic-api-key-here"
ANTHROPIC_MODEL="claude-haiku-4-5-20251001"
```

| Model | Speed | Quality | Cost |
|---|---|---|---|
| `claude-haiku-4-5-20251001` | Fast | Good — recommended for query reformulation | Low |
| `claude-sonnet-4-6` | Medium | Higher quality | Medium |
| `claude-opus-4-7` | Slow | Best quality | High |

**2. Sentence-transformers — for embedding queries into Elasticsearch KNN search**

```
EMBED_MODEL="BAAI/bge-small-en-v1.5"
```

| Model | Dimensions | Size | Quality |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | ~80 MB | Fast / lightweight |
| `BAAI/bge-small-en-v1.5` | 512 | ~120 MB | Good — default |
| `all-mpnet-base-v2` | 768 | ~420 MB | Better |
| `BAAI/bge-base-en-v1.5` | 768 | ~420 MB | Better |
| `BAAI/bge-large-en-v1.5` | 1024 | ~1.2 GB | Best local quality |

> **Important:** the embedding dimension must match your Elasticsearch KNN index. If you change `EMBED_MODEL` you must re-index all your Elasticsearch data.

The model is downloaded automatically on first run.

### How to start the toolset

Run the following command for starting the Napari GUI
```
python -m src.plugin_napari
```
On the right there is the panel control that will start or stop the MCP Microscope Toolset server.

```
Add the mcp server to you claude code account

$ claude code add --transport http microscope http://127.0.0.1:5500/mcp

Then goes in /mcp

$ /mcp + enter

And select the microscope MCP server either connecting the server or enabling the server, and from the terminal whery you started the napari-plugin you will see if the server correctly connected.
```
After you added the *mcp.json* configuration file, you can start the MCP Client that will connect to the server.

### Project structure
To add


### Execution guardrails

The `execute_python_code` tool runs agent-submitted Python code through a series of safety and correctness checks before execution:

**General checks (all code):**
- Blocks re-instantiation of `CMMCorePlus` / `UniMMCore` — the pre-configured `mmc` instance must be used
- Blocks `.loadSystemConfiguration()` / `.loadConfig()` — hardware configuration is managed by the toolset
- Blocks any reference to `viewer` or `napari.current_viewer()` — GUI access must go through the dedicated `viewer_*` tools
- Auto-detects missing packages before execution and surfaces them for user approval

**Library-specific guardrails (cellpose):**

When `cellpose` is imported the following checks are enforced:

| Check | What it blocks | Why |
|---|---|---|
| `diameter` required | Calls without an explicit diameter kwarg | Cellpose defaults to 30 px — wrong for most microscopy samples |
| `channels` required | Calls without explicit channel assignment | Default `[0,0]` fails silently on multichannel fluorescence images |
| `flow_threshold` range | Literal values outside `[0.0, 3.0]` | Values > 1.0 accept noise as cells; < 0.0 misses real cells |
| `cellprob_threshold` range | Literal values outside `[-6.0, 6.0]` | Extreme values cause silent over/under-segmentation |
| Image size (runtime) | Images larger than 512×512 px | Large images make segmentation very slow |

If an image exceeds 512×512, the agent receives an error with a ready-to-use resize + mask rescale snippet (using `cv2.INTER_NEAREST` to avoid blending label IDs). To opt out and use the original image size, set `cellpose_allow_large_image = True` in the code before the `eval` call.

**Adding guardrails for other libraries:**

The guardrail system is extensible. Static (AST-based) guards and runtime (monkey-patch) guards can be registered for any library from any module:

```python
Execute.register_library_guard("mylib", my_ast_guard_fn)   # runs before exec
Execute.register_runtime_guard("mylib", my_installer_fn)   # runs during exec
```

If you need a guardrail for a library that is not yet covered, please [open an issue or submit a PR](https://github.com/ddd42-star/microscope-toolset/issues).

### Benchmarking

The toolset includes a simulation-based benchmarking system for evaluating agent performance using the knowledge database from the *self-learn-loop*. Each benchmark test is a self-contained virtual microscope scenario served to the agent over HTTP — the agent cannot see the ground truth or simulation configuration.

Tests are launched from the **Benchmarking** panel in the MCPServer GUI, or from the CLI:

```bash
# Start a test server on port 5602
python -m src.benchmarking.test_server test_1 --port 5602

# List all available tests
python -m src.benchmarking.test_runner
```

To add a new test, see [Benchmark Tests](docs/benchmark_test_authoring.md).

---

### TO DO LIST

- [x] Fix use of Elasticsearch and PostgresSQL database
- [x] Add summary of Claude Code Agent session
- [ ] Calculate some Analysis insight as: nb tokens, duration, final code, whole conversation between user and agent. Was planning to do it on a jupyter notebook but if a better way exists then lets implement it
- [x] Add the possibility to use a remote core. This will replace the part of executing it on the mcp tool execution code, but for other image analysis, will need to stay.
- [ ] Plan the experiments to do on the real microscope to show the train & untrained Agent.
- [ ] Plan to create additional metadata from the microscope session
- [x] Build a chatbox for visualising user-agent conversation, including time, tool calls, ect.
- [ ] Switch local virtual simulation to virtual simulation from the package virtual_microscope

