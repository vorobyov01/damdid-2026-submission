#!/usr/bin/env python3
"""Is the published 34-39% FSDP peak saving an artefact of measurement order?

The original memory_experiment.py measures FSDP and single-GPU in the SAME
process, FSDP first.  This script reproduces that protocol and its mirror
image, using the identical measurement code as mem_time.py, so the two orders
can be compared directly against the isolated single-mode-per-process numbers.

  torchrun --nproc_per_node=2 order_probe.py --order fsdp_first
  torchrun --nproc_per_node=2 order_probe.py --order single_first
"""
import argparse, os, torch, torch.distributed as dist
import mem_time as mt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", choices=["fsdp_first", "single_first"], required=True)
    ap.add_argument("--h", type=int, default=4096)
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--retain", action="store_true",
                    help="keep returned bound tensors alive, as the original script does")
    args = ap.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    dist.init_process_group("nccl", rank=rank, world_size=ws)
    mt.install_comm_timers(ws)
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    torch.manual_seed(42)

    model = mt.make_mlp(784, args.h, args.d).to(dev).eval()
    x = torch.randn(1, 1, 28, 28, device=dev).clamp(0, 1)

    seq = (["fsdp", "single"] if args.order == "fsdp_first"
           else ["single", "fsdp"])
    out = {}
    for mode in seq:
        out[mode] = mt.measure(model, x, args.eps, dev, mode, ws, rank, "CROWN",
                               retain=args.retain)

    if rank == 0:
        print(f"\n===== order={args.order} retain={args.retain} "
              f"(same process) h={args.h} d={args.d} =====")
        for mode in seq:
            r = out[mode]
            print(f"  {mode:<7} params={r['params_MB']:8.1f} resident={r['resident_MB']:9.1f} "
                  f"peak={r['peak_MB']:9.1f} transient={r['transient_MB']:9.1f} "
                  f"wall={r['wall_ms']:8.1f}ms")
        s, f = out["single"]["peak_MB"], out["fsdp"]["peak_MB"]
        print(f"  reported saving = {(1 - f / s) * 100:+.1f}%  "
              f"(single {s:.1f} -> fsdp {f:.1f})")
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
