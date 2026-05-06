# Microscope Toolset — Control Panel Overview

The control panel is the central hub of the Microscope Toolset.
It is a small sidebar widget inside napari that lets you connect to a microscope,
start the AI agent server, and manage all the supporting services from one place.

---

## What it does at a glance

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   Your .cfg file                                             │
│        │                                                     │
│        ▼                                                     │
│   ┌─────────────┐     ┌──────────────────┐                  │
│   │  Microscope │────►│  napari viewer   │                  │
│   │  Core       │     │  (live image,    │                  │
│   │  (real or   │     │   channels,      │                  │
│   │   virtual)  │     │   stage control) │                  │
│   └─────────────┘     └──────────────────┘                  │
│        │                                                     │
│        ▼                                                     │
│   ┌─────────────┐     ┌──────────────────┐                  │
│   │  MCP Server │────►│  Claude Code     │                  │
│   │  (AI tools) │     │  (LLM agent)     │                  │
│   └─────────────┘     └──────────────────┘                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

In short: the control panel wires the microscope hardware (or a simulation)
to the napari GUI and to the AI agent, and keeps everything in sync.

---

## The main sections

### 1. Microscope Core

This is the most important section. Before anything else works, a core must be
loaded — it is the software representation of the microscope.

```
┌─ Microscope Core ─────────────────────────────────────────┐
│                                                           │
│  Status:  ●  CMMCorePlus   127.0.0.1:5601                │
│                                                           │
│  Remote:  [ http://192.168.1.10:5601 ]   [ Connect ]     │
│  Proxy port:  [ 5601 ]                                    │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

There are two ways to get a core running:

**A — Load a local configuration file**
Inside the napari-micromanager toolbar (the toolbar at the top of the viewer),
browse to a `.cfg` file and click **Load**.
The control panel intercepts that click, starts a small local server for the
microscope, and connects napari to it automatically.

**B — Connect to a remote server**
If the microscope is on another computer (or a colleague is sharing theirs),
paste its URL into the Remote field and click **Connect**.
The control panel connects to that server and hands the same napari interface
to it, so the experience is identical.

The coloured dot shows the state at a glance:

| Colour | Meaning |
|--------|---------|
| Grey | No core loaded |
| Orange | Starting or switching |
| Green | Ready |
| Red | Error |

The small badge next to the dot (`CMM+`, `Uni`, or `RMC`) tells you which kind
of core is running — real hardware, virtual simulation, or remote.

---

#### How switching configurations works

When you load a different `.cfg` file, the control panel checks whether the
new file needs a different type of core (real hardware vs. virtual simulation).

```
Same type of core?
  YES ──► reload in place, no disruption to the UI
  NO  ──► stop the old server → start a fresh one → reconnect the UI
```

This is handled automatically. The napari interface always reflects whatever
core is currently active.

---

### 2. MCP Server

The MCP (Model Context Protocol) server exposes the microscope to Claude Code
as a set of AI tools.  Once the core is ready, click **Start** to launch it.

```
┌─ MCP Server ──────────────────────────────────────────────┐
│                                                           │
│  MCP:  [ 127.0.0.1 ] : [ 5500 ]                          │
│  ●  MCP Server   http://127.0.0.1:5500   [ Stop ]        │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

Claude Code connects to this URL and can then snap images, move the stage,
run acquisitions, and analyse results — all through natural language.

The MCP server restarts automatically whenever the microscope core changes
(e.g. after loading a new configuration), so the agent always has a fresh
connection to the active hardware.

---

### 3. Elasticsearch  ·  PostgreSQL  _(optional)_

These panels appear only if the relevant service is configured in the `.env` file.

- **Elasticsearch** — powers the semantic search used by the database agent
  (API documentation, device manuals, scientific papers).
- **PostgreSQL** — stores session logs so the agent can recall past experiments.

Both are independent of the microscope core and can be started or stopped at any
time without affecting the running acquisition.

---

### 4. Benchmarking

Lets you launch a pre-built virtual microscope scenario (a "test server") to
evaluate the agent's performance without touching real hardware.

```
┌─ Benchmarking ────────────────────────────────────────────┐
│                                                           │
│  Test: [ blood_smear — Fluorescence cell ]  Port: [5602] │
│  ●  Test Server   http://127.0.0.1:5602   [ Stop ]       │
│                                                           │
│  Blood Smear  |  Channels: DAPI, FITC                    │
│  Simulated fluorescence with synthetic cells             │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

When a test server is launched, the agent is automatically connected to it as
if it were a real microscope, so a full experiment cycle can be benchmarked
end-to-end.

---

### 5. Experiment Tracking

Records the full timeline of an experiment — all agent tool calls, images,
and results — into a local workspace folder.

```
┌─ Experiment Tracking ─────────────────────────────────────┐
│                                                           │
│  ●  Name: [ my_experiment ]  [ Start Tracking ]          │
│  Tracking: my_experiment_2026…  |  workspace: …/runs/…   │
│                                                   [ Open ]│
│                                                           │
└───────────────────────────────────────────────────────────┘
```

Click **Open** at any time to browse the saved workspace in the file explorer.

---

### 6. Status bar

A single line at the bottom of the panel always shows what is happening right
now.  Colour indicates severity:

| Colour | Meaning |
|--------|---------|
| Blue | Waiting for user action |
| Orange | Something is starting or loading |
| Green | Ready |
| Red | An error occurred |

---

## Startup flow

This is what happens from the moment you open napari to the moment the agent
is ready to use:

```
napari opens
     │
     ▼
Control panel appears as a dock widget
     │
     ▼  (after ~500 ms)
napari-micromanager toolbar is added at the top of the viewer
     │
     ▼
User browses to a .cfg file and clicks Load
     │
     ├── If same core type as before ──► reload in the existing server
     │
     └── If different core type (or first load)
              │
              ▼
         Local proxy server starts  (real hardware or virtual simulation)
              │
              ▼  (server accepts connections)
         napari UI reconnects to the new core
         Channels, presets, shutters, stages all repopulate
              │
              ▼
         MCP Server can now be started
              │
              ▼
         Claude Code connects → agent is ready
```

---

## Configuration via `.env`

A few values can be set in the `.env` file to pre-configure the panel for a
shared lab setup:

| Variable | What it does |
|----------|-------------|
| `PROXY_CORE_HOST` | Pre-fills the Remote URL field with a specific machine's address |
| `PROXY_CORE_PORT` | Pre-fills the proxy port on both the Remote and Proxy port fields |
| `CFGPATH` | Loads this `.cfg` file automatically on startup |
| `ELASTICSEARCH` | Enables the Elasticsearch panel |
| `DB_HOST` | Enables the PostgreSQL panel |

Values set manually in the UI are remembered across sessions and take priority
over the `.env` defaults.
