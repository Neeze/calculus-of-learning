# Chạy thí nghiệm trên Google Colab

Các cell dưới đây copy nguyên vào Colab là chạy. Thứ tự: **Setup → (Drive) → E1/E2**.

- **Runtime GPU** (T4/L4/A100): dùng `setup_colab.sh`.
- **Runtime TPU** (v5e-1): dùng `setup_colab_tpu.sh`. E1 Tầng 1 (DreamerV3) và E2 đều là JAX nên chạy TPU tốt; ước tính v5e-1: ~1–2h/run DreamerV3 `size1m` 500k steps (đo fps thật ở Cell 5b trước khi tin).

---

## Cell 1 — Clone code

`third_party/dreamerv3` là git submodule — dùng `--recurse-submodules` để kéo cả nó trong một lệnh (script setup ở Cell 2 cũng tự `git submodule update --init` nếu bạn quên cờ này, nên không bắt buộc phải nhớ chính xác):

```bash
%%bash
git clone --recurse-submodules https://github.com/Neeze/calculus-of-learning.git
```

```python
%cd calculus-of-learning
```

## Cell 2 — Setup môi trường

**GPU runtime:**
```bash
!bash scripts/collab/setup_colab.sh
```

**TPU runtime (v5e-1):**
```bash
!bash scripts/collab/setup_colab_tpu.sh
```

Sau đó set biến môi trường cho notebook (mọi cell chạy thí nghiệm đều cần):

```python
import os
os.environ["MUJOCO_GL"] = "osmesa"   # GPU runtime: "egl"
# TPU runtime only:
os.environ["JAX_PLATFORMS"] = "tpu"
```

> Nếu import lỗi sau setup: Runtime → Restart session, chạy lại từ Cell 2 (bỏ Cell 1).

## Cell 3 — (Khuyến nghị) Mount Drive để giữ checkpoint qua các phiên

Colab ngắt phiên là mất disk. Trỏ `outputs/` vào Drive để DreamerV3 **tự resume** và kết quả không mất:

```python
from google.colab import drive
drive.mount('/content/drive')

import os
persist = '/content/drive/MyDrive/calculus_outputs'
os.makedirs(persist, exist_ok=True)
!rm -rf outputs && ln -s {persist} outputs
```

---

## Cell 4 — E1 Tầng 0 (toy, có ground truth) — ~30 phút, chạy được cả CPU

```bash
!bash scripts/run_e1_error_laws.sh
```

Kết quả: `outputs/plots/`, bảng in ra stdout.

## Cell 5 — E1 Tầng 1 (DreamerV3 size1m trên DMC)

Spec: `docs/E1-Tier1-DreamerV3-spec.md`. Giai đoạn A (train 3 task × 5 seeds) chạy bằng entry point gốc của DreamerV3.

### Cell 5a — Train một run (~1–2h TPU v5e / A10; tự resume nếu đứt phiên)

```bash
%%bash
cd third_party/dreamerv3
python dreamerv3/main.py \
  --logdir ../../outputs/dreamer/dmc_walker_walk/seed0 \
  --configs dmc_proprio \
  --task dmc_walker_walk \
  --seed 0 \
  --run.steps 5e5 \
  --jax.platform tpu   # GPU runtime: bỏ dòng này
```

### Cell 5b — Đo fps trước khi cam kết cả sweep (10–15 phút)

Chạy Cell 5a khoảng 15 phút rồi dừng, đọc số `fps` trong log:
tổng thời gian 1 run ≈ `500000 / fps` giây. Nhân 15 runs để lên kế hoạch phiên.

### Cell 5c — Sweep 1 task × 5 seeds (~1 phiên Colab / task)

```bash
%%bash
cd third_party/dreamerv3
TASK=dmc_walker_walk    # đổi: dmc_cartpole_swingup, dmc_cheetah_run
for SEED in 0 1 2 3 4; do
  python dreamerv3/main.py \
    --logdir ../../outputs/dreamer/${TASK}/seed${SEED} \
    --configs dmc_proprio --task ${TASK} \
    --seed ${SEED} --run.steps 5e5 \
    --jax.platform tpu   # GPU runtime: bỏ dòng này
done
```

Chạy lại cell sau khi đứt phiên là tiếp tục từ checkpoint (nhờ logdir trên Drive).

> Giai đoạn B–D (probe rollout `E(h)`, distill block predictor, analyze) — code trong
> `experiments/e1_error_laws/dreamer/` theo spec §7; **chưa implement**, sẽ bổ sung.
> Train xong giai đoạn A trước là đúng thứ tự — B–D chỉ cần checkpoint, chạy sau được.

## Cell 6 — E2 Frame Freedom vs OOD (~1–2h toàn bộ)

Pipeline đầy đủ: collect (cache datasets) → sanity oracle → train 5 config × 3 seeds → eval → analyze:

```bash
!bash scripts/run_e2_frame_freedom.sh --env walker-walk
!bash scripts/run_e2_frame_freedom.sh --env cheetah-run
```

- Đọc kỹ output bước `[2/5]` (shift-strength oracle) — nếu có `[WARN]` thì dừng, xem spec `docs/E2-Frame-Freedom-vs-OOD-Generalization.md` §6.
- Kết quả: `outputs/results/e2/` (`results_*.json`, `stats_*.json`, 4 bảng CSV) và `outputs/plots/` (hình E2-A, E2-B).

## Cell 7 — Đóng gói kết quả (nếu không dùng Drive)

```bash
!zip -r results.zip outputs/results outputs/plots
from google.colab import files; files.download('results.zip')
```

---

## Tóm tắt thời gian (ước tính — đo lại bằng Cell 5b)

| Thí nghiệm | Thiết bị | Thời gian |
|---|---|---|
| E1 Tầng 0 (toy) | CPU/bất kỳ | ~30 phút |
| E1 Tầng 1, 1 run | TPU v5e-1 / A10 | ~1–2h |
| E1 Tầng 1, đủ 15 runs | TPU v5e-1 | ~20–30h (chia 3 phiên theo task) |
| E2, 1 env (15 runs) | bất kỳ có JAX | ~1h |
