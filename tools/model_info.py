#!/usr/bin/env python3
"""Print key model config facts for debugging."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/public-nfs/wlx/models/Qwen/Qwen3___6-27B"
c = json.load(open(f"{path}/config.json"))
tc = c.get("text_config", c)

lt = tc.get("layer_types", [])
fa = sum(1 for x in lt if x == "full_attention")
la = sum(1 for x in lt if x == "linear_attention")

print(f"model_type:          {c.get('model_type')}")
print(f"num_hidden_layers:   {tc.get('num_hidden_layers')}")
print(f"full_attention:      {fa}")
print(f"linear_attention:    {la}")
print(f"head_dim:            {tc.get('head_dim')}")
print(f"hidden_size:         {tc.get('hidden_size')}")
print(f"num_attn_heads:      {tc.get('num_attention_heads')}")
print(f"num_kv_heads:        {tc.get('num_key_value_heads')}")
print(f"architectures:       {c.get('architectures')}")
