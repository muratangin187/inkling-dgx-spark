"""Materialize process-local OpenCode or isolated Pi settings."""

import json
import os
from pathlib import Path
import sys

mode, config_dir, api = sys.argv[1:]
config_dir = Path(config_dir)

if mode in ("opencode", "opencode-low"):
    config = json.loads(os.environ.get("OPENCODE_CONFIG_CONTENT") or "{}")
    profile = json.loads((config_dir / "opencode.json").read_text())
    provider = profile["providers"]["inkling-dgx"]
    provider["settings"]["baseURL"] = api
    if mode == "opencode-low":
        provider["models"]["inkling-2-mini"]["body"]["reasoning_effort"] = "low"
    config["model"] = profile["model"]
    config.setdefault("providers", {})["inkling-dgx"] = provider
    config.setdefault("agents", {}).setdefault("title", {})["disabled"] = True
    print(json.dumps(config))
elif mode == "pi":
    state = Path(os.environ["PI_CODING_AGENT_DIR"])
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    models = json.loads((config_dir / "pi-models.json").read_text())
    models["providers"]["inkling-dgx"]["baseUrl"] = api
    (state / "models.json").write_text(json.dumps(models, indent=2) + "\n")
    settings = state / "settings.json"
    if not settings.exists():
        settings.write_text((config_dir / "pi-settings.json").read_text())
else:
    raise ValueError(f"Unsupported client profile: {mode}")
