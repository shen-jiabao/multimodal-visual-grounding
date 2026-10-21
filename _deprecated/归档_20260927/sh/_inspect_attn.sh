#!/bin/bash
set +e
D=/root/.cache/huggingface/modules/transformers_modules/LocateAnything-3B
echo "################# modeling_locateanything.py : __init__ / attn / language_model 构造 #################"
grep -nE "attn|language_model|def __init__|AutoModel|Qwen2|from_config|_attn_implementation" "$D/modeling_locateanything.py" | head -60
echo
echo "################# Qwen2 forward 注意力分支（1300-1345）#################"
sed -n '1300,1345p' "$D/modeling_qwen2.py"
echo
echo "################# Qwen2 支持的 attn 字符串（grep）#################"
grep -nE "_attn_implementation ==|in \(|flash_attention_2|sdpa|eager|ALL_ATTENTION" "$D/modeling_qwen2.py" | head -30