import os
import numpy as np
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from pytransform3d import rotations as pr

def _quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                     w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], dtype=np.float64)

def normalize_quat_xyzw(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=float)
    if pr.quaternion_requires_renormalization(q):
        q = q / np.linalg.norm(q)
    if q[-1] < 0:
        q = -q
    return q

def plot_loss_curves_with_dual_style(hist_train, hist_val, save_list, ckpt_dir):

    epochs = list(range(1, len(hist_train) + 1))

    def _plot(include_all_saves: bool, fname: str):
        plt.figure(figsize=(12, 7))
        plt.plot(epochs, hist_train, marker='o', markersize=4, label='Train Loss')
        plt.plot(epochs, hist_val,   marker='v', markersize=4, label='Val Loss')

        # 标保存点的竖线
        if include_all_saves:
            for e, flag in zip(epochs, save_list):
                if flag:
                    plt.axvline(x=e, linestyle='--', linewidth=1, color='gray')
        else:
            if any(save_list):
                last_e = [e for e, f in zip(epochs, save_list) if f][-1]
                plt.axvline(x=last_e, linestyle='--', linewidth=1, color='gray')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        os.makedirs(ckpt_dir, exist_ok=True)
        out_path = os.path.join(ckpt_dir, fname)
        plt.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
        plt.close()

    _plot(include_all_saves=False, fname="loss_curve_fin.png")
    _plot(include_all_saves=True,  fname="loss_curve_debug.png")