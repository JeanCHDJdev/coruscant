#!/usr/bin/env bash

set -euo pipefail

show_usage() {
    cat <<'EOF'
Usage: bash setup.sh

Register the default coruscant Jupyter kernel inside the current pixi environment.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            show_usage >&2
            exit 1
            ;;
    esac
done

echo "Registering Jupyter kernel..."
env -u PYTHONPATH -u PYTHONSTARTUP -u PYTHONHOME python -m ipykernel install --user --name coruscant --display-name coruscant

echo "Success. Enter the environment with: pixi shell"
echo "Available project tasks: pixi task list"