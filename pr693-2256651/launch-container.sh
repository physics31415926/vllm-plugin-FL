#!/usr/bin/env bash
set -euo pipefail

run_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$run_dir"
exec >driver.log 2>&1

inside=${1:-inside.sh}
case "$inside" in
  inside.sh|adaptation-gate.sh|benchmark.sh) ;;
  *) exit 2 ;;
esac
container="codex-$(basename "$run_dir" | tr '[:upper:]' '[:lower:]')"
expected=2256651ac575dfacce82dce376f6a5256a4210ca
image=harbor.baai.ac.cn/plugin/vllm-plugin-fl@sha256:734c6f7d620e2ca3be4fa382a9e2cd6e1d345a310e2e502f11317b5521b886ae
model_dir=${MODEL_DIR:?set MODEL_DIR to the local models directory}
export GEMS_VENDOR=metax VLLM_PLUGINS=fl

cleanup() {
  status=$?
  if [[ -n "${container_owned:-}" ]]; then
    docker rm -f "$container" >/dev/null 2>&1 || true
  fi
  printf '%s\n' "$status" >"$run_dir/exit"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

exec 9>"${LOCK_PATH:-$run_dir/serving.lock}"
flock -n 9
printf 'Started %s\n' "$(date -Is)"
test -r "$model_dir/Qwen3.6-27B/model.safetensors.index.json"
test -r "$model_dir/Qwen3.6-35B-A3B/model.safetensors.index.json"
docker image inspect "$image" --format '{{.Id}} {{.RepoDigests}}'
git init --quiet src
printf '%s\n' "$expected" > src/.git/shallow
git -C src bundle unbundle "$run_dir/pr693-2256651.bundle"
git -C src checkout --quiet --detach "$expected"
test "$(git -C src rev-parse HEAD)" = "$expected"
test -z "$(git -C src status --porcelain)"
printf 'Source %s\n' "$expected"
mx-smi | tee gpu-before.log
grep -Fq 'no process found' gpu-before.log

docker create --name "$container" --pull=never --hostname vllm-plugin-fl \
  --ipc=host --shm-size=64g \
  --device /dev/dri:/dev/dri:rwm --device /dev/mxcd:/dev/mxcd:rwm \
  --env GEMS_VENDOR --env VLLM_PLUGINS \
  --mount "type=bind,source=$run_dir,target=/qualification" \
  --mount "type=bind,source=$model_dir/Qwen3-0.6B,target=/data/models/Qwen/Qwen3-0.6B,readonly" \
  --mount "type=bind,source=$model_dir/Qwen3.6-27B,target=/data/models/Qwen/Qwen3.6-27B,readonly" \
  --mount "type=bind,source=$model_dir/Qwen3.6-35B-A3B,target=/data/models/Qwen/Qwen3.6-35B-A3B,readonly" \
  --workdir /qualification/src "$image" bash "/qualification/$inside"
container_owned=1
docker start -a "$container" || true
exit "$(docker inspect --format '{{.State.ExitCode}}' "$container")"
