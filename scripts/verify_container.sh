#!/usr/bin/env bash
# Run on the receiving computer. Model downloads and real inference are required.
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-cpu}"
compose=(docker compose -f docker-compose.yml)
quant=()
case "$mode" in
  cpu) ;;
  gpu) compose+=(-f docker-compose.gpu.yml); quant=(--load-4bit) ;;
  *) echo "Usage: bash scripts/verify_container.sh [cpu|gpu]" >&2; exit 2 ;;
esac
command -v docker >/dev/null || { echo "Install Docker with Compose v2 first." >&2; exit 1; }
mkdir -p outputs
"${compose[@]}" config --quiet
"${compose[@]}" build
"${compose[@]}" run --rm -T vlm-fo1 python cli.py --check --json "outputs/${mode}-preflight.json"
"${compose[@]}" run --rm -T vlm-fo1 python inference.py "${quant[@]}" \
  --max-new-tokens 16 --output "outputs/${mode}-acceptance.jpg" --json "outputs/${mode}-acceptance.json"
"${compose[@]}" run --rm -T vlm-fo1 python - "$mode" <<'PY'
import json, sys
from pathlib import Path
from PIL import Image
mode = sys.argv[1]
p = json.loads(Path(f'outputs/{mode}-acceptance.json').read_text())
assert p['model']['decoder_checkpoint'] == 'Qwen/Qwen3.5-0.8B', p['model']
assert p['proposal_source'] == 'user' and p['num_proposals'] == 7
assert p['raw_output'].strip(), 'Generation returned no text'
assert p['bridge_is_trained'] is False and p['limitations']
assert p['inference_status']['backend'] == 'fo1_hfre_qwen35'
for item in p['detections']:
    assert 0 <= item['region_index'] < p['num_proposals']
    assert item['bbox'] == p['proposals'][item['region_index']]
with Image.open(p['output_image']) as image:
    image.verify()
print('PASS: real container inference, Qwen3.5 decoder, JSON and image output.')
print('This check establishes execution only. Grounding quality is not validated.')
PY
