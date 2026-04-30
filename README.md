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

### Create the vector database with pdf files
To help the different agents to avoid hallucination, it's advised to create a vector database with the different "knowledge". We have the documentation of *pymmcore_plus* and the publications of the *Pertz Lab*. If you want to add other pdfs file you can run this command:
```
python .\src\create_database_from_publication.py --db <path to db> --doc <path to pdf(s)>
```
In the folder that you choose were to save the database, at the moment two directories will be created: *pages_png* and *pages_markdown*. The first will contain the png files of each page of the document and in the second the markdown files of the text extracted. 

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
Example of claude code

$ claude code add --transport http microscope http://127.0.0.1:5500/mcp

Then goes in /mcp

$ /mcp + enter

And select the microscope MCP server either connecting the server or enabling the server, and from the terminal whery you started the napari-plugin you will see if the server correctly connected.
```
After you added the *mcp.json* configuration file, you can start the MCP Client that will connect to the server.


### TO DO LIST

- [ ] Fix use of Elasticsearch and PostgresSQL database
- [x] Add summary of Claude Code Agent session
- [ ] Calculate some Analysis insight as: nb tokens, duration, final code, whole conversation between user and agent. Was planning to do it on a jupyter notebook but if a better way exists then lets implement it
- [ ] Add the possibility to use a remote core. This will replace the part of executing it on the mcp tool execution code, but for other image analysis, will need to stay.
- [ ] Plan the experiments to do on the real microscope to show the train & untrained Agent.
- [ ] Plan to create additional metadata from the microscope session
- [x] Build a chatbox for visualising user-agent conversation, including time, tool calls, ect.
- [ ] Switch local virtual simulation to virtual simulation from the package virtual_microscope

