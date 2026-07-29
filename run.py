"""
run.py

The only file the user runs directly. Shows a numbered menu, and
runs whichever model's own infer script, using that model's own venv.

Each model's infer script is called with NO arguments -- its input
and output folder paths are hardcoded inside that script itself
(pointing at the shared common_input/ folder and that model's own
models/<n>/output/ folder). This file doesn't pass any paths.

Usage:
    python run.py            -> shows menu, prompts for a number
    python run.py --model 1  -> skips the prompt, runs model 1 directly
"""

import argparse
import os
import subprocess
import sys

from model_registry import MODELS


def print_menu() -> None:
    print("\nAvailable models:")
    for number, info in MODELS.items():
        print(f"  [{number}] {info['label']}")
    print()


def build_env(info: dict) -> dict:
    """
    Most models (Python venvs) need no special environment -- the venv's
    own python.exe already knows where its packages live. The Lua Torch7
    model is the exception: luajit.exe needs LUA_PATH/LUA_CPATH set (same
    as setpaths.cmd does interactively) since we're launching it directly
    via subprocess, skipping that manual step. This merges any such
    per-model overrides from model_registry.py's optional "env" key into
    a copy of the current environment.
    """
    env = os.environ.copy()
    extra = info.get("env", {})

    # PATH_PREPEND is a convention (not a real env var) meaning "put this
    # folder at the front of PATH", used so the model's own DLLs are found
    # before anything else on the system with the same name.
    prepend = extra.get("PATH_PREPEND")
    if prepend:
        env["PATH"] = prepend + os.pathsep + env.get("PATH", "")

    for key, value in extra.items():
        if key != "PATH_PREPEND":
            env[key] = value

    return env


def run_model(number: str) -> None:
    if number not in MODELS:
        print(f"'{number}' is not a valid choice. Valid options: {', '.join(MODELS)}")
        sys.exit(1)

    info = MODELS[number]
    python_exe = info["python"]
    script = info["script"]

    if not python_exe.exists():
        print(f"Error: interpreter not found at {python_exe}")
        print(f"Have you set up the environment for '{info['label']}' yet?")
        sys.exit(1)

    if not script.exists():
        print(f"Error: infer script not found at {script}")
        print("Update the 'script' path in model_registry.py to match where your infer script actually lives.")
        sys.exit(1)

    print(f"\nRunning [{number}] {info['label']} ...")
    print(f"Script: {script}\n")

    env = build_env(info)

    # cwd=script.parent so the script's own relative paths (if any) resolve
    # the same way they would if you ran it directly from its own folder.
    subprocess.run([str(python_exe), str(script)], check=True, cwd=script.parent, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model number to run directly, skipping the menu.")
    args = parser.parse_args()

    if args.model:
        run_model(args.model)
        return

    print_menu()
    choice = input("Enter the number of the model to run: ").strip()
    run_model(choice)


if __name__ == "__main__":
    main()
