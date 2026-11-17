"""Upload weights, results and data that are too large for git to the Hugging Face Hub.
Token is read from the HF environment variable (never stored in the repo). Usage: upload_hf.py [small|big|all]"""
import os, sys
from huggingface_hub import HfApi

what = sys.argv[1] if len(sys.argv) > 1 else "all"
api = HfApi(endpoint="https://huggingface.co", token=os.environ["HF"])
user = api.whoami()["name"]
M, D = f"{user}/flash-lewm", f"{user}/flash-lewm-data"
api.create_repo(M, repo_type="model", private=True, exist_ok=True)
api.create_repo(D, repo_type="dataset", private=True, exist_ok=True)
if what in ("small", "all"):
    api.upload_folder(repo_id=M, repo_type="model", folder_path="ckpt", path_in_repo="ckpt", allow_patterns=["*.ckpt", "*.pt", "*.npz"])
    api.upload_folder(repo_id=M, repo_type="model", folder_path="results", path_in_repo="results", allow_patterns=["*.json"])
    api.upload_folder(repo_id=M, repo_type="model", folder_path="logs", path_in_repo="logs", allow_patterns=["*.log"])
    print("model repo done", flush=True)
    for src, dst in [("/data1/cgh/dataset/cube_eval/eval_pairs.npz", "eval_pairs/cube_eval_pairs.npz"),
                     ("/data1/cgh/dataset/cube_eval/eval_pairs_v2.npz", "eval_pairs/cube_eval_pairs_v2.npz"),
                     ("/data1/cgh/dataset/doom/eval_pairs.npz", "eval_pairs/doom_eval_pairs.npz"),
                     ("/data1/cgh/dataset/doom/eval_pairs500.npz", "eval_pairs/doom_eval_pairs500.npz"),
                     ("/data1/cgh/dataset/ant/eval_pairs.npz", "eval_pairs/ant_eval_pairs.npz")]:
        api.upload_file(repo_id=D, repo_type="dataset", path_or_fileobj=src, path_in_repo=dst); print("uploaded", dst, flush=True)
if what in ("big", "all"):
    api.upload_file(repo_id=D, repo_type="dataset", path_or_fileobj="/data1/cgh/dataset/doom/mwh_all.npz", path_in_repo="doom/mwh_all.npz")
    print("uploaded doom/mwh_all.npz", flush=True)
print("ALL DONE", flush=True)
