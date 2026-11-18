"""Upload every resource that does not live in git: weights, evaluation episodes, converted data, rollouts, results, logs.

Targets
  Hugging Face (private): model repo <user>/flash-lewm (ckpt/, results/, logs/) and dataset repo <user>/flash-lewm-data.
  GitHub Release `assets-v1` of GuihaiChen1998/flash-lewm: every file up to 2 GB (GitHub's per-asset limit), flat names.
Only files missing at the target (or with a different size) are sent, so the script can be re-run after an interruption.
Tokens come from the environment (HF, GH) and are never written to the repo.
Usage: HF=... GH=... python scripts/upload_all.py [hf|gh|all]
Needs HF_ENDPOINT=https://huggingface.co HF_HUB_DISABLE_XET=1 behind the Linux#1 proxy."""
import glob, os, sys, time
import requests
from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = "/data1/cgh/dataset"
os.chdir(ROOT)
what = sys.argv[1] if len(sys.argv) > 1 else "all"

# (local path, path in the HF repo, repo kind)  -- model repo: weights; dataset repo: data
def files():
    out = []
    for f in sorted(glob.glob("ckpt/**/*", recursive=True)):
        if os.path.isfile(f) and os.path.basename(f) != "environment.json":   # environment.json holds the training shell's env vars
            out.append((f, f, "model"))
    data = [(f"{DS}/cube_eval/eval_pairs.npz", "eval_pairs/cube_eval_pairs.npz"),
            (f"{DS}/cube_eval/eval_pairs_v2.npz", "eval_pairs/cube_eval_pairs_v2.npz"),
            (f"{DS}/doom/eval_pairs.npz", "eval_pairs/doom_eval_pairs.npz"),
            (f"{DS}/doom/eval_pairs500.npz", "eval_pairs/doom_eval_pairs500.npz"),
            (f"{DS}/ant/eval_pairs.npz", "eval_pairs/ant_eval_pairs.npz"),
            (f"{DS}/reacher_eval/eval_pairs.npz", "eval_pairs/reacher_eval_pairs.npz"),
            (f"{DS}/libero/study1.npz", "libero/study1.npz"),
            (f"{DS}/libero/libero10.npz", "libero/libero10.npz"),
            (f"{DS}/doom/mwh_all.npz", "doom/mwh_all.npz"),
            (f"{DS}/official_h5/doom_mwh.h5", "official_h5/doom_mwh.h5"),
            (f"{DS}/official_h5/libero10.h5", "official_h5/libero10.h5")]
    data += [(f, "libero/" + os.path.basename(f)) for f in sorted(glob.glob(f"{DS}/libero/pairs*"))]
    data += [(f, "rollouts/" + os.path.basename(f)) for f in sorted(glob.glob(f"{DS}/rollouts/*.npz"))]
    out += [(a, b, "dataset") for a, b in data if os.path.exists(a)]
    return out


def hf():
    api = HfApi(endpoint="https://huggingface.co", token=os.environ["HF"])
    user = api.whoami()["name"]
    repo = {"model": f"{user}/flash-lewm", "dataset": f"{user}/flash-lewm-data"}
    for k, r in repo.items():
        api.create_repo(r, repo_type=k, private=True, exist_ok=True)
    # results/ and logs/ are many small files: one commit each, only changed files are sent
    for folder, pat in [("results", ["*.json"]), ("logs", ["*.log"])]:
        api.upload_folder(repo_id=repo["model"], repo_type="model", folder_path=folder, path_in_repo=folder, allow_patterns=pat)
        print("synced", folder, flush=True)
    have = {k: {s.rfilename: s.size for s in api.repo_info(r, repo_type=k, files_metadata=True).siblings} for k, r in repo.items()}
    for src, dst, k in files():
        if have[k].get(dst) == os.path.getsize(src):
            continue
        for attempt in range(5):
            try:
                api.upload_file(repo_id=repo[k], repo_type=k, path_or_fileobj=src, path_in_repo=dst)
                print("hf", k, dst, os.path.getsize(src), flush=True); break
            except Exception as e:
                print("retry", dst, repr(e)[:200], flush=True); time.sleep(10)
    print("HF DONE", flush=True)


def gh():
    repo, tag = "GuihaiChen1998/flash-lewm", "assets-v1"
    h = {"Authorization": f"Bearer {os.environ['GH']}", "Accept": "application/vnd.github+json"}
    rel = requests.get(f"https://api.github.com/repos/{repo}/releases/tags/{tag}", headers=h).json()
    assets, page = {}, 1
    while True:
        a = requests.get(f"https://api.github.com/repos/{repo}/releases/{rel['id']}/assets?per_page=100&page={page}", headers=h).json()
        if not a:
            break
        assets.update({x["name"]: x["size"] for x in a}); page += 1
    for src, dst, k in files():
        size = os.path.getsize(src)
        flat = dst.startswith("eval_pairs/") or (dst.startswith("ckpt/") and dst.count("/") == 1)
        name = os.path.basename(dst) if flat else dst.replace("/", "__")     # flat names match the assets uploaded in round 11
        if size > 2_000_000_000 or assets.get(name) == size:
            continue
        for attempt in range(5):
            try:
                with open(src, "rb") as f:
                    r = requests.post(f"https://uploads.github.com/repos/{repo}/releases/{rel['id']}/assets?name={name}",
                                      headers={**h, "Content-Type": "application/octet-stream"}, data=f, timeout=3600)
                print("gh", name, size, r.status_code, flush=True)
                if r.status_code in (201, 422):
                    break
            except Exception as e:
                print("retry", name, repr(e)[:200], flush=True); time.sleep(10)
    print("GH DONE", flush=True)


if what in ("hf", "all"):
    hf()
if what in ("gh", "all"):
    gh()
print("ALL DONE", flush=True)
