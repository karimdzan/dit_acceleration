import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_glob", type=str, required=True, help='e.g. "runs/sweep/N*/metrics.json"')
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.results_glob))
    if not paths:
        raise SystemExit(f"No metrics found for glob: {args.results_glob}")

    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            m = json.load(f)
        parts = Path(p).parts
        n = None
        for s in parts:
            if s.startswith("N"):
                try:
                    n = int(s[1:])
                except:
                    pass
        if n is None:
            n = int(m.get("train_size", -1))
        rows.append((n, m))

    rows.sort(key=lambda x: x[0])
    Ns = [r[0] for r in rows]

    pfd_gen = [r[1].get("pfd_teacher_student_mean") for r in rows]
    fid_gen = [r[1].get("fid_student_teacher") for r in rows]

    pfd_mem = [r[1].get("pfd_student_empirical_mean") for r in rows]
    mdist   = [r[1].get("m_distance_student_to_train_mean") for r in rows]
    fid_mem = [r[1].get("fid_student_train") for r in rows]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.xscale("log")
    plt.plot(Ns, pfd_gen, marker="o")
    plt.xlabel("Train size N (log scale)")
    plt.ylabel("PFD(student, teacher)  (generalization proxy)")
    plt.title("Generalization vs N")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "generalization_pfd.png", dpi=150)
    plt.close()

    plt.figure()
    plt.xscale("log")
    plt.plot(Ns, fid_gen, marker="o")
    plt.xlabel("Train size N (log scale)")
    plt.ylabel("FID(student, teacher)  (generalization proxy)")
    plt.title("Generalization vs N")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "generalization_fid.png", dpi=150)
    plt.close()

    plt.figure()
    plt.xscale("log")
    plt.plot(Ns, pfd_mem, marker="o")
    plt.xlabel("Train size N (log scale)")
    plt.ylabel("PFD(student, empirical)  (memorization proxy)")
    plt.title("Memorization vs N")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "memorization_pfd.png", dpi=150)
    plt.close()

    plt.figure()
    plt.xscale("log")
    plt.plot(Ns, mdist, marker="o")
    plt.xlabel("Train size N (log scale)")
    plt.ylabel("M-distance (NN dist to train)  (memorization baseline)")
    plt.title("Memorization vs N")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "memorization_mdistance.png", dpi=150)
    plt.close()

    plt.figure()
    plt.xscale("log")
    plt.plot(Ns, fid_mem, marker="o")
    plt.xlabel("Train size N (log scale)")
    plt.ylabel("FID(student, train)  (memorization-ish proxy)")
    plt.title("Memorization vs N")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "memorization_fid_train.png", dpi=150)
    plt.close()

    print(f"Ok: wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()
