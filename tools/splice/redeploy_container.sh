#!/usr/bin/env bash
# One-command recovery: recreate the GPU container and re-apply the full splice.
set -u
exec > >(tee -a ~/redeploy.log) 2>&1
echo "=== redeploy $(date -u) ==="

sudo docker rm -f treeval 2>/dev/null
sudo docker run -d --name treeval --gpus all --shm-size 8g \
  -v ~/sglang:/fork --entrypoint sleep lmsysorg/sglang:latest infinity
sleep 4

X() { sudo docker exec treeval bash -lc "$1"; }

echo "== container GPU =="
X "nvidia-smi -L | head -1" || { echo "FATAL container has no GPU"; exit 1; }

SITE=$(X 'python3 -c "import sglang,os;print(os.path.dirname(sglang.__file__))"' | tr -d '\r')
echo "site: $SITE"

echo "== overlay fork additive files =="
X "cp -rf /fork/python/sglang/srt/tree $SITE/srt/ && \
   cp -f /fork/python/sglang/srt/entrypoints/openai/protocol_tree.py $SITE/srt/entrypoints/openai/ && \
   cp -f /fork/python/sglang/srt/entrypoints/openai/serving_tree.py $SITE/srt/entrypoints/openai/"

echo "== install runtime + apply splice =="
sudo docker cp ~/splice/tree_runtime.py treeval:$SITE/srt/tree/tree_runtime.py
sudo docker cp ~/splice/apply_splice.py treeval:/tmp/apply_splice.py
sudo docker cp ~/splice/fix_hooks.py treeval:/tmp/fix_hooks.py
X "python3 /tmp/apply_splice.py $SITE && python3 /tmp/fix_hooks.py $SITE"

echo "== fix tree serving registration (chat serving, not template manager) =="
X "python3 - << 'PYEOF'
import re
p = '$SITE/srt/entrypoints/http_server.py'
s = open(p).read()
bad = '''    fast_api_app.state.openai_serving_tree = OpenAIServingTree(  # [autotree-splice]
        _global_state.tokenizer_manager, _global_state.template_manager
    )
'''
if bad in s:
    s = s.replace(bad, '')
    anchor = '    fast_api_app.state.openai_serving_chat = '
    idx = s.index(anchor); le = s.index('\n    )', idx) + len('\n    )') + 1
    inject = '''    fast_api_app.state.openai_serving_tree = OpenAIServingTree(  # [autotree-splice]
        _global_state.tokenizer_manager, fast_api_app.state.openai_serving_chat
    )
'''
    s = s[:le] + inject + s[le:]
    open(p,'w').write(s); print('registration fixed')
else:
    print('registration already correct or differs')
PYEOF"

echo "== import check =="
X "python3 -c 'import sglang.srt.managers.scheduler, sglang.srt.managers.tokenizer_manager, sglang.srt.entrypoints.http_server; print(\"IMPORTS_OK\")'"

echo "== boot server =="
sudo docker exec -d treeval bash -c "exec python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct --port 30000 --mem-fraction-static 0.6 > /tmp/server.log 2>&1"
for i in $(seq 1 40); do
  sleep 10
  if X "curl -sf http://127.0.0.1:30000/health" >/dev/null 2>&1; then echo "SERVER READY after $((i*10))s"; break; fi
  [ "$i" = "40" ] && { echo "FATAL server never ready"; X "tail -5 /tmp/server.log"; exit 1; }
done
echo "=== REDEPLOY DONE $(date -u) ==="
