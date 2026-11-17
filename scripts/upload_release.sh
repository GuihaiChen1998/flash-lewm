#!/bin/bash
# Upload weights and evaluation episodes as GitHub Release assets (too large for git history).
# Token is read from the GH environment variable. Usage: GH=... scripts/upload_release.sh
set -u
REPO=GuihaiChen1998/flash-lewm; TAG=assets-v1; API=https://api.github.com/repos/$REPO
AUTH="Authorization: Bearer $GH"
id=$(curl -sS -H "$AUTH" $API/releases/tags/$TAG | python3 -c "import sys,json;print(json.load(sys.stdin).get('id',''))")
if [ -z "$id" ]; then
  id=$(curl -sS -H "$AUTH" -X POST $API/releases -d "{\"tag_name\":\"$TAG\",\"name\":\"Weights and evaluation episodes\",\"body\":\"See docs/HANDOFF.md for what each file is.\"}" | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")
fi
echo release $id
up() {  # up <file> <asset name>
  code=$(curl -sS -o /dev/null -w '%{http_code}' -H "$AUTH" -H "Content-Type: application/octet-stream" --data-binary @"$1" "https://uploads.github.com/repos/$REPO/releases/$id/assets?name=$2")
  echo "$2 $code"
}
for f in ckpt/*.ckpt ckpt/*.pt; do up $f $(basename $f); done
up /data1/cgh/dataset/cube_eval/eval_pairs.npz cube_eval_pairs.npz
up /data1/cgh/dataset/cube_eval/eval_pairs_v2.npz cube_eval_pairs_v2.npz
up /data1/cgh/dataset/doom/eval_pairs.npz doom_eval_pairs.npz
up /data1/cgh/dataset/doom/eval_pairs500.npz doom_eval_pairs500.npz
up /data1/cgh/dataset/ant/eval_pairs.npz ant_eval_pairs.npz
echo ALL DONE
