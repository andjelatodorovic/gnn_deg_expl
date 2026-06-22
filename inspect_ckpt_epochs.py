import torch, glob, os

paths = glob.glob(r"storage/checkpoints/**/id_best.ckpt", recursive=True)
paths = sorted(paths, key=os.path.getmtime, reverse=True)

for p in paths[:10]:
    print("\n", p)
    ckpt = torch.load(p, map_location="cpu")
    print("keys:", ckpt.keys())
    if "epoch" in ckpt:
        print("epoch =", ckpt["epoch"])
    if "config" in ckpt:
        cfg = ckpt["config"]
        print("config type:", type(cfg))
        try:
            print("train.max_epoch =", cfg.train.max_epoch)
            print("train.mile_stones =", cfg.train.mile_stones)
        except Exception as e:
            print("could not read config train fields:", e)
