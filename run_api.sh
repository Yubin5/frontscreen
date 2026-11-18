#!/usr/bin/env bash
# Run the screen flattening API (POST /flatten) and the SAM/detector server it needs.
#
#   ./run_api.sh            start (or restart) the API, starting the SAM/detector server first if needed
#   ./run_api.sh stop       stop the API container (the SAM/detector container is left running)
#   ./run_api.sh status     container state + /health
#
# Images: docker build -t frontscreen-api .   and   docker build -t frontscreen-sam sam/   (CPU).
# Containers use host networking and mount this folder, so code edits need only a restart.
# Machine-specific settings go in local.env (not tracked), e.g.
#   API_HOSTS=127.0.0.1,10.0.0.5     extra bind address (off-loopback callers then need secrets/api_key)
#   SAM_IMAGE=my-xpu-torch-image SAM_DEVICE=xpu SAM_DOCKER_ARGS="--device /dev/dri --group-add render"
set -euo pipefail
cd "$(dirname "$0")"
LAB="$PWD"
[ -f local.env ] && . ./local.env
API_HOSTS="${API_HOSTS:-127.0.0.1}"
API_PORT="${API_PORT:-8793}"
API_IMAGE="${API_IMAGE:-frontscreen-api}"
API_NAME="${API_NAME:-frontscreen-api}"
SAM_IMAGE="${SAM_IMAGE:-frontscreen-sam}"
SAM_NAME="${SAM_NAME:-frontscreen-sam}"
SAM_DEVICE="${SAM_DEVICE:-cpu}"
SAM_DOCKER_ARGS="${SAM_DOCKER_ARGS:-}"

running() { [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = "true" ]; }

start_sam() {
  if running "$SAM_NAME"; then return; fi
  if docker inspect "$SAM_NAME" >/dev/null 2>&1; then
    docker start "$SAM_NAME" >/dev/null
  else
    # shellcheck disable=SC2086
    docker run -d --name "$SAM_NAME" --restart unless-stopped --network host $SAM_DOCKER_ARGS \
      -e DEVICE="$SAM_DEVICE" -v "$LAB:/lab" -v "$LAB/cache/hf:/root/.cache/huggingface" -w /lab/sam \
      "$SAM_IMAGE" python sam_server.py >/dev/null
  fi
  for _ in $(seq 1 90); do curl -sf http://127.0.0.1:8791/health >/dev/null && return; sleep 2; done
  echo "SAM/detector server did not come up (docker logs $SAM_NAME)" >&2; exit 1
}

start_api() {
  docker rm -f "$API_NAME" >/dev/null 2>&1 || true
  docker run -d --name "$API_NAME" --restart unless-stopped --network host \
    -e HOSTS="$API_HOSTS" -e PORT="$API_PORT" -e PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    -v "$LAB:/lab" -v "$LAB/cache/paddlex:/root/.paddlex" -w /lab --entrypoint python "$API_IMAGE" api_server.py >/dev/null
  for _ in $(seq 1 90); do curl -sf "http://127.0.0.1:$API_PORT/health" >/dev/null && break; sleep 1; done
}

case "${1:-start}" in
  start)  start_sam; start_api; curl -s "http://127.0.0.1:$API_PORT/health"; echo ;;
  stop)   docker rm -f "$API_NAME" >/dev/null && echo "$API_NAME stopped" ;;
  status) for c in "$SAM_NAME" "$API_NAME"; do printf '%-18s %s\n' "$c" "$(docker inspect -f '{{.State.Status}}' "$c" 2>/dev/null || echo missing)"; done
          curl -s "http://127.0.0.1:$API_PORT/health"; echo ;;
  *) echo "usage: $0 [start|stop|status]" >&2; exit 2 ;;
esac
