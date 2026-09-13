#!/bin/bash
# Real-serving benchmark against a running llama-server, using the /completion
# endpoint's own returned `timings` block (prompt_per_second /
# predicted_per_second) -- this is what a real deployment actually
# experiences (full sampler chain, detokenization, response construction),
# unlike llama-bench's tg/pp numbers which call llama_decode on a constant
# dummy token with no sampling at all.
#
# Usage: bench_server.sh <host:port> <reps> <n_predict> ["sampler json fragment"]
#   sampler json fragment example for near-greedy/minimal-sampler-cost:
#     '"temperature":0,"dry_multiplier":0,"xtc_probability":0,"top_k":1'

set -euo pipefail
ADDR="${1:-localhost:8090}"
REPS="${2:-5}"
N_PREDICT="${3:-64}"
SAMPLER_JSON="${4:-}"
PROMPT="${5:-The capital of France is a city with a rich history dating back over two thousand years, and today it stands as}"

echo "Server: $ADDR | reps=$REPS n_predict=$N_PREDICT sampler_extra={${SAMPLER_JSON}}"
echo "---"

pp_sum=0
tg_sum=0
for i in $(seq 1 "$REPS"); do
  body=$(python3 -c "
import json
extra = {}
sampler_json = '''${SAMPLER_JSON}'''
if sampler_json.strip():
    extra = json.loads('{' + sampler_json + '}')
payload = {'prompt': '''${PROMPT}''', 'n_predict': ${N_PREDICT}, 'cache_prompt': False, 'timings_per_token': False}
payload.update(extra)
print(json.dumps(payload))
")
  resp=$(curl -s -X POST "http://${ADDR}/completion" -H 'Content-Type: application/json' -d "$body")
  pp=$(echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['timings']['prompt_per_second'])" 2>/dev/null || echo "ERR")
  tg=$(echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['timings']['predicted_per_second'])" 2>/dev/null || echo "ERR")
  echo "run $i: prompt_per_second=$pp  predicted_per_second=$tg"
  if [ "$pp" != "ERR" ]; then
    pp_sum=$(python3 -c "print($pp_sum + $pp)")
    tg_sum=$(python3 -c "print($tg_sum + $tg)")
  else
    echo "  RAW RESPONSE: $resp" | head -c 500
  fi
done

echo "---"
python3 -c "print(f'mean prompt_per_second = {$pp_sum/$REPS:.2f}')"
python3 -c "print(f'mean predicted_per_second = {$tg_sum/$REPS:.2f}')"
