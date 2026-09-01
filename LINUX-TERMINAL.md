# Linux / Chromebook Terminal Control

This project can be controlled entirely from a Linux terminal. The `xau` command controls the existing GitHub-hosted company and its temporary hosted dashboard; your Chromebook does not need to stay awake after startup.

## First-time setup on Chromebook

1. Enable **Linux development environment** in ChromeOS Settings.
2. Open the Linux Terminal.
3. Install Git and clone the repository:

```bash
sudo apt update
sudo apt install -y git
git clone https://github.com/lafaillejeremiah7-droid/XAUUSD-Company.git
cd XAUUSD-Company
bash scripts/install-linux.sh
```

4. Authenticate GitHub once:

```bash
gh auth login
```

Choose GitHub.com, HTTPS, and browser authentication when prompted.

5. Reload your shell so `xau` is on PATH:

```bash
source ~/.bashrc
```

## Everyday commands

```bash
xau status
```
Shows the authoritative ON/OFF control plus recent runtime workflow jobs.

```bash
xau start
```
Turns the company ON. If control was OFF, this updates `runtime-control.json`, increments its generation, commits to `main`, and pushes. If control is already ON but no runtime is active, it dispatches a fresh runtime session.

```bash
xau dashboard
```
Prints the current hosted dashboard URL after the runtime publishes it.

```bash
xau dashboard --open
```
Opens the hosted dashboard in the Linux browser when `xdg-open` is available.

```bash
xau stop
```
Turns the authoritative control OFF first, pushes that state to `main`, then cancels runtime sessions that existed before the OFF commit. This closes the hosted dashboard tunnel with the runtime.

```bash
xau logs
```
Shows the latest runtime workflow. Full downloaded logs are shown after the workflow completes.

```bash
xau update
```
Fast-forwards the local checkout to the latest `main`.

```bash
xau test
```
Runs the local Python compile check and complete pytest suite using `.venv`.

```bash
xau local-dashboard
```
Runs the dashboard locally at `http://127.0.0.1:8080`. This is for local UI development/preview and is separate from the hosted runtime.

## Safety behavior

- `xau start` and `xau stop` refuse to modify the control file when your local checkout has uncommitted changes.
- `xau stop` writes OFF to GitHub before cancelling the active job, so killing a runner cannot leave the authoritative control logically ON.
- The hosted trading company remains signal-only; it does not execute brokerage orders.
- Telegram secrets remain in GitHub Actions repository secrets. The CLI does not download or print them.
- `.env` is created only for optional local/manual runs and should never be committed.

## Updating the controller

```bash
cd ~/XAUUSD-Company
xau update
bash scripts/install-linux.sh
```

The installer is idempotent and keeps the `xau` command linked to this repository checkout.
