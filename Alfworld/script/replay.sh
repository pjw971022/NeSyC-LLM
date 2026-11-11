#!/bin/bash
# Reproduce every number reported in the README from the shipped LLM transcripts, without an API key.
# Each run's success rate is printed next to the reported one; the script exits non-zero on any mismatch.
# All runs start in parallel (about 40 minutes on 20 cores). usage: bash Alfworld/script/replay.sh [name-filter]
cd "$(dirname "$0")/../.." || exit 1
unset OPENAI_API_KEY OPENAI_BASE_URL  # a cache miss must fail, not call an API
D=./Alfworld/data; OUT=$(mktemp -d); FILTER=${1:-}
F=$D/llm_cache_claude-fable-5-2.json.gz; FM=$D/llm_cache_claude-fable-5-2_react_memory.json.gz
H=$D/llm_cache_claude-haiku-5-5.json.gz; O=$D/llm_cache_claude-opus-5-5.json.gz
O2=$D/llm_cache_claude-opus-5-5_val2.json.gz; O3=$D/llm_cache_claude-opus-5-5_val3.json.gz
V1="--hard $D/hard_stream_test.json"; V2="--hard $D/hard_stream_test2.json"; V3="--hard $D/hard_stream_test3.json"
RUNS=(  # name | reported success rate | transcript | model | script | extra arguments
  "fable_static_123|0.9919|$F|claude-fable-5-2|static|"
  "fable_20_nesyc|1.0000|$F|claude-fable-5-2|static|--eval_episode_num 20"
  "fable_20_react|1.0000|$F|claude-fable-5-2|static|--eval_episode_num 20 --method react"
  "fable_20_reflexion|1.0000|$F|claude-fable-5-2|static|--eval_episode_num 20 --method reflexion"
  "fable_hard_nesyc|1.0000|$F|claude-fable-5-2|hard|"
  "fable_hard_react|1.0000|$F|claude-fable-5-2|hard|--method react"
  "fable_hard_reflexion|1.0000|$F|claude-fable-5-2|hard|--method reflexion"
  "fable_stream_nesyc|0.3636|$F|claude-fable-5-2|hard_stream|"
  "fable_stream_react|0.0000|$F|claude-fable-5-2|hard_stream|--method react"
  "fable_stream_react_memory|0.9091|$FM|claude-fable-5-2|hard_stream|--method react_memory"
  "fable_stream_reflexion|1.0000|$F|claude-fable-5-2|hard_stream|--method reflexion"
  "haiku_20_nesyc|0.9500|$H|claude-haiku-5-5|static|--eval_episode_num 20"
  "haiku_20_react|1.0000|$H|claude-haiku-5-5|static|--eval_episode_num 20 --method react"
  "haiku_20_reflexion|1.0000|$H|claude-haiku-5-5|static|--eval_episode_num 20 --method reflexion"
  "haiku_suite_nesyc|0.0000|$H|claude-haiku-5-5|hard_suite|"
  "haiku_suite_react|0.0000|$H|claude-haiku-5-5|hard_suite|--method react"
  "haiku_suite_reflexion|0.0000|$H|claude-haiku-5-5|hard_suite|--method reflexion"
  "opus_dev_nesyc|0.0000|$O|claude-opus-5-5|hard_stream|"
  "opus_dev_revise|0.1818|$O|claude-opus-5-5|hard_stream|--revise"
  "opus_dev_hints|0.0000|$O|claude-opus-5-5|hard_stream|--hints"
  "opus_dev_hints_revise|0.0000|$O|claude-opus-5-5|hard_stream|--hints --revise"
  "opus_dev_plus|0.0000|$O|claude-opus-5-5|hard_stream|--plus"
  "opus_val1_nesyc|0.0000|$O|claude-opus-5-5|hard_stream|$V1"
  "opus_val1_revise|0.0000|$O|claude-opus-5-5|hard_stream|$V1 --revise"
  "opus_val2_nesyc|0.0000|$O2|claude-opus-5-5|hard_stream|$V2"
  "opus_val2_hints|0.0000|$O2|claude-opus-5-5|hard_stream|$V2 --hints"
  "opus_val2_hints_revise|0.0000|$O2|claude-opus-5-5|hard_stream|$V2 --hints --revise"
  "opus_val3_nesyc|0.0000|$O3|claude-opus-5-5|hard_stream|$V3"
  "opus_val3_plus|0.7000|$O3|claude-opus-5-5|hard_stream|$V3 --plus"
)
for run in "${RUNS[@]}"; do
  IFS='|' read -r name want cache engine script extra <<< "$run"
  [[ -n $FILTER && $name != *$FILTER* ]] && continue
  # shellcheck disable=SC2086  # $extra holds several arguments
  LLM_CACHE=$cache ENGINE=$engine bash Alfworld/script/$script.sh $extra \
    --rule_path "$OUT/$name.rules" --save_path "$OUT/$name.csv" > "$OUT/$name.log" 2>&1 &
done
wait
status=0
for run in "${RUNS[@]}"; do
  IFS='|' read -r name want _ <<< "$run"
  [[ -n $FILTER && $name != *$FILTER* ]] && continue
  got=$(grep -oE 'Average SR: [0-9.]+' "$OUT/$name.log" | tail -1 | cut -d' ' -f3)
  if [[ $got == "$want" ]] && ! grep -q Traceback "$OUT/$name.log"; then mark=ok; else mark=MISMATCH; status=1; fi
  printf '%-28s reported %s  replayed %-7s %s\n' "$name" "$want" "${got:--}" "$mark"
done
echo "logs: $OUT"
exit $status
