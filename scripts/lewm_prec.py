
import sys, time, torch
sys.path.insert(0, ".")
from flash_lewm.lewm import load_lewm, lewm_preds
m = load_lewm("ckpt/lewm/pusht")
g = torch.Generator(device="cuda").manual_seed(0)
z = torch.randn(1, 192, device="cuda"); zg = torch.randn(1, 192, device="cuda")
pl = torch.randn(1, 300, 5, 10, device="cuda", generator=g)
def run():
    return (lewm_preds(m, z, pl) - zg.view(1,1,1,-1)).pow(2).sum(-1)
def t(fn, k=30):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s = time.time()
    for _ in range(k): fn()
    torch.cuda.synchronize(); return (time.time()-s)/k*1000
print(torch.cuda.get_device_name())
ref = run()
print("fp32 1 rollout ms", t(run))
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
c = run(); print("tf32 ms", t(run), "rel", ((c-ref).abs()/ref.abs()).max().item(), "rank-agree top30", len(set(c[0,:,-1].topk(30,largest=False).indices.tolist()) & set(ref[0,:,-1].topk(30,largest=False).indices.tolist())))
def runb():
    with torch.autocast("cuda", dtype=torch.bfloat16): return run().float()
c = runb(); print("bf16 ms", t(runb), "rel", ((c-ref).abs()/ref.abs()).max().item(), "top30 agree", len(set(c[0,:,-1].topk(30,largest=False).indices.tolist()) & set(ref[0,:,-1].topk(30,largest=False).indices.tolist())))
