# XAU/USD Animated Company Dashboard

The dashboard is a read-only visualization layer. It does not make or modify trading decisions.

## Run it

Start the company normally in one terminal:

```bash
python main.py
```

Then start the dashboard server in a second terminal:

```bash
python dashboard_server.py
```

Open `http://127.0.0.1:8080`.

For a remote/container host, bind deliberately:

```bash
python dashboard_server.py --host 0.0.0.0 --port 8080
```

## What the figures mean

Each miniature business figure has a truthful state: idle, working, passing a document, receiving a document, complete, or blocked/vetoed. The paper flights are generated from runtime milestones observed from the existing `xau-company` logger. The performance panel reads the persistent outcome ledger directly.

Runtime dashboard state is written atomically to `data/dashboard_state.json` and is ignored by Git.
