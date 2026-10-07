#!/usr/bin/env bash
# End-to-end smoke test against the full docker compose stack.
#
# Without trained weights (app/classifier/models/classifier.pt absent) it checks:
#   - Vault seeding, migrations and the Casbin policy seed
#   - api and worker REFUSE to start (weights missing)
#   - admin bootstrap through the CLI
#   - a TIFF dropped over SFTP becomes a batch + document + queued job
# With weights it additionally checks the full path:
#   - api healthy, admin login, SFTP drop classified and visible in GET /batches/{id}
#     within the 10 s end-to-end budget
set -euo pipefail

cd "$(dirname "$0")/.."
[ -f .env ] || cp .env.example .env
COMPOSE="docker compose"
SFTP_SECRET=docclassifier_dev_pw  # dev credential of the local sftp container
ADMIN_EMAIL=admin@example.com
ADMIN_SECRET="smoke-$(date +%s)-Secret"
HAVE_WEIGHTS=false
[ -f app/classifier/models/classifier.pt ] && HAVE_WEIGHTS=true

fail() { echo "SMOKE FAIL: $*" >&2; $COMPOSE ps -a >&2 || true; exit 1; }
psql_q() { $COMPOSE exec -T db psql -U docclassifier -d docclassifier -tAc "$1"; }
wait_for() {  # wait_for <seconds> <description> <command...>
  local deadline=$((SECONDS + $1)); local what=$2; shift 2
  until "$@" >/dev/null 2>&1; do
    [ $SECONDS -ge $deadline ] && fail "timed out waiting for: $what"
    sleep 1
  done
}

# Start from a clean slate, as CI always does. Leftover volumes break reruns: an admin created
# by an earlier run keeps its old password (so login fails), and a volume made by an older
# image can be unwritable. NOTE: this deletes this compose project's containers and volumes.
echo "== reset compose project (containers and volumes)"
$COMPOSE down -v --remove-orphans >/dev/null 2>&1 || true

echo "== build and start infrastructure"
$COMPOSE build
$COMPOSE up -d db redis minio sftp vault
wait_for 90 "db healthy" sh -c "$COMPOSE ps db | grep -q healthy"
wait_for 90 "vault healthy" sh -c "$COMPOSE ps vault | grep -q healthy"

echo "== seed Vault and migrate"
$COMPOSE run --rm vault-init >/dev/null
$COMPOSE run --rm migrate
[ "$(psql_q "select count(*) from casbin_rule where ptype='p'")" -gt 0 ] || fail "policy table empty"

echo "== bootstrap admin"
$COMPOSE run --rm -e ADMIN_SECRET="$ADMIN_SECRET" migrate python -m app.cli create-admin --email "$ADMIN_EMAIL"
[ "$(psql_q "select count(*) from casbin_rule where ptype='g' and v1='admin'")" = "1" ] || fail "admin role not granted"

echo "== start sftp-ingest"
$COMPOSE up -d sftp-ingest

if [ "$HAVE_WEIGHTS" = false ]; then
  echo "== no weights: api and worker must refuse to start"
  $COMPOSE up -d --no-deps api worker || true
  wait_for 60 "api to exit" sh -c "$COMPOSE ps -a api | grep -q Exited"
  wait_for 60 "worker to exit" sh -c "$COMPOSE ps -a worker | grep -q Exited"
  $COMPOSE logs api | grep -q "Classifier weights missing" || fail "api did not report missing weights"
  $COMPOSE logs worker | grep -q "refused_to_start" || fail "worker did not refuse to start"
else
  echo "== start api and worker"
  $COMPOSE up -d api worker
  wait_for 120 "api healthy" sh -c "$COMPOSE ps api | grep -q healthy"
fi

echo "== drop a TIFF over SFTP"
$COMPOSE run --rm --no-deps -v "$PWD/scripts:/scripts:ro" -e SFTP_SECRET="$SFTP_SECRET" \
  migrate python /scripts/sftp_upload.py --host sftp --name smoke.tif
START=$SECONDS
wait_for 20 "batch recorded" sh -c "[ \"\$($COMPOSE exec -T db psql -U docclassifier -d docclassifier -tAc 'select count(*) from document')\" -ge 1 ]"
[ "$(psql_q "select action from audit_log where action='batch.created' limit 1")" = "batch.created" ] || fail "no batch.created audit entry"

if [ "$HAVE_WEIGHTS" = false ]; then
  QUEUED=$($COMPOSE exec -T redis redis-cli LLEN rq:queue:classify | tr -d '\r')
  [ "$QUEUED" -ge 1 ] || fail "no classification job queued"
  echo "SMOKE OK (pipeline up to queued job; train the model for the full path)"
  exit 0
fi

echo "== full path: prediction visible through the api"
TOKEN=$(curl -fsS -X POST "localhost:${API_PORT:-8000}/auth/jwt/login" \
  --data-urlencode "username=$ADMIN_EMAIL" --data-urlencode "password=$ADMIN_SECRET" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')
BATCH=$(psql_q "select id from batch order by created_at desc limit 1")
wait_for 10 "batch classified" sh -c "curl -fsS -H 'Authorization: Bearer $TOKEN' localhost:${API_PORT:-8000}/batches/$BATCH | grep -q '\"status\":\"done\"'"
echo "end-to-end latency: $((SECONDS - START)) s (budget 10 s)"
curl -fsS -H "Authorization: Bearer $TOKEN" "localhost:${API_PORT:-8000}/batches/$BATCH" | python3 -m json.tool
echo "SMOKE OK (full path)"
