#!/bin/sh
# Startup is side-effect free. The CLI/demo loads weights only when needed;
# an automatic UPN checkpoint download additionally requires UPN_AUTO_DOWNLOAD=1.
set -eu

if [ "$#" -eq 0 ]; then
    set -- python cli.py --help
fi

exec "$@"
