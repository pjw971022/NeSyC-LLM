#!/bin/bash
# Reproduce every number reported in the README from the shipped LLM transcripts, without an API key.
# Each run's success rate is printed next to the reported one; the script exits non-zero on any mismatch.
# All runs start in parallel (about an hour on 20 cores). usage: bash Alfworld/script/replay.sh [name-filter]
cd "$(dirname "$0")/../.." || exit 1
unset OPENAI_API_KEY OPENAI_BASE_URL  # a cache miss must fail, not call an API
D=./Alfworld/data; OUT=$(mktemp -d); FILTER=${1:-}
F=$D/llm_cache_claude-fable-5-2.json.gz; FM=$D/llm_cache_claude-fable-5-2_react_memory.json.gz
H=$D/llm_cache_claude-haiku-5-5.json.gz; O=$D/llm_cache_claude-opus-5-5.json.gz
O2=$D/llm_cache_claude-opus-5-5_val2.json.gz; O3=$D/llm_cache_claude-opus-5-5_val3.json.gz
O4=$D/llm_cache_claude-opus-5-5_val4.json.gz; O4P=$D/llm_cache_claude-opus-5-5_val4_plus.json.gz; O4V=$D/llm_cache_claude-opus-5-5_val4_v2.json.gz
SA=$D/llm_cache_claude-sonnet-5-5_paper.json.gz; SP=$D/llm_cache_claude-sonnet-5-5_plus.json.gz; SV=$D/llm_cache_claude-sonnet-5-5_v2.json.gz
HA=$D/llm_cache_claude-haiku-5-5_paper.json.gz; HP=$D/llm_cache_claude-haiku-5-5_plus.json.gz; HS=$D/llm_cache_claude-haiku-5-5_suite.json.gz; HV=$D/llm_cache_claude-haiku-5-5_v2.json.gz
V1="--hard $D/hard_stream_test.json"; V2="--hard $D/hard_stream_test2.json"; V3="--hard $D/hard_stream_test3.json"
V4="--split train --hard $D/hard_stream_test4.json"
V5="--split train --hard $D/hard_stream_test5"  # + a, b or c
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
  "sonnet_val3_nesyc|0.0000|$SA|claude-sonnet-5-5|hard_stream|$V3"
  "sonnet_val3_plus|0.0000|$SP|claude-sonnet-5-5|hard_stream|$V3 --plus"
  "haiku_val3_nesyc|0.0000|$HA|claude-haiku-5-5|hard_stream|$V3"
  "haiku_val3_plus|0.0000|$HP|claude-haiku-5-5|hard_stream|$V3 --plus"
  "haiku_suite_plus|0.0000|$HS|claude-haiku-5-5|hard_suite|--plus"
  "haiku_suite_v2|0.5000|$HS|claude-haiku-5-5|hard_suite|--v2"
  "opus_dev_v2|0.4545|$O4|claude-opus-5-5|hard_stream|--v2"
  "opus_val4_nesyc|0.0000|$O4|claude-opus-5-5|hard_stream|$V4"
  "opus_val4_plus|0.9000|$O4P|claude-opus-5-5|hard_stream|$V4 --plus"
  "opus_val4_v2|0.7000|$O4V|claude-opus-5-5|hard_stream|$V4 --v2"
  "sonnet_val4_nesyc|0.0000|$SA|claude-sonnet-5-5|hard_stream|$V4"
  "sonnet_val4_plus|0.0000|$SP|claude-sonnet-5-5|hard_stream|$V4 --plus"
  "sonnet_val4_v2|0.8000|$SV|claude-sonnet-5-5|hard_stream|$V4 --v2"
  "haiku_val4_nesyc|0.0000|$HA|claude-haiku-5-5|hard_stream|$V4"
  "haiku_val4_plus|0.9000|$HP|claude-haiku-5-5|hard_stream|$V4 --plus"
  "haiku_val4_v2|0.0000|$HV|claude-haiku-5-5|hard_stream|$V4 --v2"
  "opus_val5a_nesyc|0.0000|$D/llm_cache_claude-opus-5-5_val5_paper.json.gz|claude-opus-5-5|hard_stream|${V5}a.json"
  "opus_val5b_nesyc|0.0000|$D/llm_cache_claude-opus-5-5_val5_paper.json.gz|claude-opus-5-5|hard_stream|${V5}b.json"
  "opus_val5c_nesyc|0.0000|$D/llm_cache_claude-opus-5-5_val5_paper.json.gz|claude-opus-5-5|hard_stream|${V5}c.json"
  "opus_val5a_plus|0.5000|$D/llm_cache_claude-opus-5-5_val5_plus.json.gz|claude-opus-5-5|hard_stream|${V5}a.json --plus"
  "opus_val5b_plus|0.5000|$D/llm_cache_claude-opus-5-5_val5_plus.json.gz|claude-opus-5-5|hard_stream|${V5}b.json --plus"
  "opus_val5c_plus|0.0000|$D/llm_cache_claude-opus-5-5_val5_plus.json.gz|claude-opus-5-5|hard_stream|${V5}c.json --plus"
  "opus_val5a_v2|0.9000|$D/llm_cache_claude-opus-5-5_val5_v2.json.gz|claude-opus-5-5|hard_stream|${V5}a.json --v2"
  "opus_val5b_v2|0.0000|$D/llm_cache_claude-opus-5-5_val5_v2.json.gz|claude-opus-5-5|hard_stream|${V5}b.json --v2"
  "opus_val5c_v2|0.9000|$D/llm_cache_claude-opus-5-5_val5_v2.json.gz|claude-opus-5-5|hard_stream|${V5}c.json --v2"
  "sonnet_val5a_nesyc|0.0000|$D/llm_cache_claude-sonnet-5-5_val5_paper.json.gz|claude-sonnet-5-5|hard_stream|${V5}a.json"
  "sonnet_val5b_nesyc|0.0000|$D/llm_cache_claude-sonnet-5-5_val5_paper.json.gz|claude-sonnet-5-5|hard_stream|${V5}b.json"
  "sonnet_val5c_nesyc|0.0000|$D/llm_cache_claude-sonnet-5-5_val5_paper.json.gz|claude-sonnet-5-5|hard_stream|${V5}c.json"
  "sonnet_val5a_plus|0.0000|$D/llm_cache_claude-sonnet-5-5_val5_plus.json.gz|claude-sonnet-5-5|hard_stream|${V5}a.json --plus"
  "sonnet_val5b_plus|0.0000|$D/llm_cache_claude-sonnet-5-5_val5_plus.json.gz|claude-sonnet-5-5|hard_stream|${V5}b.json --plus"
  "sonnet_val5c_plus|0.0000|$D/llm_cache_claude-sonnet-5-5_val5_plus.json.gz|claude-sonnet-5-5|hard_stream|${V5}c.json --plus"
  "sonnet_val5a_v2|0.0000|$D/llm_cache_claude-sonnet-5-5_val5_v2.json.gz|claude-sonnet-5-5|hard_stream|${V5}a.json --v2"
  "sonnet_val5b_v2|0.9000|$D/llm_cache_claude-sonnet-5-5_val5_v2.json.gz|claude-sonnet-5-5|hard_stream|${V5}b.json --v2"
  "sonnet_val5c_v2|0.9000|$D/llm_cache_claude-sonnet-5-5_val5_v2.json.gz|claude-sonnet-5-5|hard_stream|${V5}c.json --v2"
  "haiku_val5a_nesyc|0.0000|$D/llm_cache_claude-haiku-5-5_val5_paper.json.gz|claude-haiku-5-5|hard_stream|${V5}a.json"
  "haiku_val5b_nesyc|0.0000|$D/llm_cache_claude-haiku-5-5_val5_paper.json.gz|claude-haiku-5-5|hard_stream|${V5}b.json"
  "haiku_val5c_nesyc|0.0000|$D/llm_cache_claude-haiku-5-5_val5_paper.json.gz|claude-haiku-5-5|hard_stream|${V5}c.json"
  "haiku_val5a_plus|0.0000|$D/llm_cache_claude-haiku-5-5_val5_plus.json.gz|claude-haiku-5-5|hard_stream|${V5}a.json --plus"
  "haiku_val5b_plus|0.0000|$D/llm_cache_claude-haiku-5-5_val5_plus.json.gz|claude-haiku-5-5|hard_stream|${V5}b.json --plus"
  "haiku_val5c_plus|1.0000|$D/llm_cache_claude-haiku-5-5_val5_plus.json.gz|claude-haiku-5-5|hard_stream|${V5}c.json --plus"
  "haiku_val5a_v2|0.0000|$D/llm_cache_claude-haiku-5-5_val5_v2.json.gz|claude-haiku-5-5|hard_stream|${V5}a.json --v2"
  "haiku_val5b_v2|0.8000|$D/llm_cache_claude-haiku-5-5_val5_v2.json.gz|claude-haiku-5-5|hard_stream|${V5}b.json --v2"
  "haiku_val5c_v2|1.0000|$D/llm_cache_claude-haiku-5-5_val5_v2.json.gz|claude-haiku-5-5|hard_stream|${V5}c.json --v2"
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
