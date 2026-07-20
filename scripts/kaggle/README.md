# Chạy thí nghiệm trên Kaggle Notebooks

Các cell dưới đây copy nguyên vào một Kaggle Notebook. Ba biến thể accelerator có
script setup riêng — chọn đúng file cho accelerator đã bật trong sidebar.

| Accelerator (Settings → Accelerator) | Script | Ghi chú |
|---|---|---|
| TPU v5e-8 | `setup_kaggle_tpu_v5e8.sh` | 8 chip TPU, tốt cho E1 Tầng 1 (JAX) |
| GPU T4 x2 | `setup_kaggle_2xt4.sh` | 2 GPU CUDA riêng biệt, chạy song song 2 seed |
| GPU RTX 6000 | `setup_kaggle_rtx6000.sh` | 1 GPU VRAM lớn, tăng batch_size thay vì song song |

**Trước khi chạy bất kỳ cell nào:** Settings (sidebar phải) → **Internet: On**. Kaggle
tắt Internet mặc định — thiếu bước này thì `pip`/`git clone` lỗi ngay từ Cell 1.

---

## Khác biệt quan trọng so với Colab

- **Không có ổ đĩa bền như Drive.** `/kaggle/working` chỉ giữ được trong phiên hiện
  tại; khi notebook dừng, mọi thứ ngoài **Output** (kết quả "Save Version" tạo ra)
  bị xoá. DreamerV3 tự resume từ checkpoint trong `logdir` — nhưng chỉ resume được
  **trong cùng một phiên chạy liên tục**, không qua các lần restart notebook.
  Muốn giữ checkpoint qua nhiều phiên: định kỳ `tar` outputs và **Save Version**
  (Cell 8), hoặc đẩy checkpoint lên một Kaggle Dataset riêng (Cell 8b).
- **Giới hạn phiên** (tham khảo, Kaggle có thể đổi): GPU tối đa ~12h liên tục /
  ~30h/tuần theo tài khoản; TPU tối đa ~9h liên tục / ~20h/tuần. Chia sweep theo
  task như Cell 5c của bản Colab, kiểm quota trong Settings → Accelerator.
- **Không có `%cd` cross-cell tự nhiên như Colab** — mọi cell `%%bash` chạy từ
  `/kaggle/working`, đảm bảo `cd calculus-of-learning` ở đầu mỗi cell `%%bash` mới
  nếu không dùng `%cd` (cell Python) trước đó.

---

## Cell 1 — Clone code

`third_party/dreamerv3` là git submodule — `--recurse-submodules` kéo cả nó trong
một lệnh (script setup ở Cell 2 cũng tự `git submodule update --init` nếu bạn quên
cờ này, nên không bắt buộc phải nhớ chính xác):

```bash
%%bash
cd /kaggle/working
git clone --recurse-submodules https://github.com/Neeze/calculus-of-learning.git
```

```python
%cd /kaggle/working/calculus-of-learning
```

## Cell 2 — Setup môi trường (chọn đúng script cho accelerator)

```bash
!bash scripts/kaggle/setup_kaggle_tpu_v5e8.sh    # TPU v5e-8
# !bash scripts/kaggle/setup_kaggle_2xt4.sh      # GPU T4 x2
# !bash scripts/kaggle/setup_kaggle_rtx6000.sh   # GPU RTX 6000
```

Set biến môi trường cho notebook (mọi cell chạy thí nghiệm đều cần):

```python
import os
os.environ["MUJOCO_GL"] = "osmesa"   # T4x2 / RTX 6000: "egl"
os.environ["JAX_PLATFORMS"] = "tpu"  # chỉ set trên TPU
```

> Nếu import lỗi sau setup: Restart session (nút phía trên) rồi chạy lại từ Cell 2
> (bỏ Cell 1, code đã clone rồi).

---

## Cell 3 — E1 Tầng 0 (toy, có ground truth) — ~30 phút, chạy được cả CPU

```bash
!bash scripts/run_e1_error_laws.sh
```

## Cell 4 — E1 Tầng 1 (DreamerV3 size1m trên DMC)

Spec: `docs/E1-Tier1-DreamerV3-spec.md`.

### Cell 4a — Đo fps trước khi cam kết cả sweep (10–15 phút)

```bash
%%bash
cd /kaggle/working/calculus-of-learning/third_party/dreamerv3
timeout 900 python dreamerv3/main.py \
  --logdir ../../outputs/dreamer/dmc_walker_walk/seed0 \
  --configs dmc_proprio --task dmc_walker_walk --seed 0 --run.steps 5e5 \
  --jax.platform tpu   # T4x2/RTX 6000: --jax.platform cuda
```

Đọc số `fps` trong log; tổng thời gian 1 run ≈ `500000 / fps` giây.

### Cell 4b — Sweep, tận dụng nhiều device song song

**TPU v5e-8 (8 chip):** một process JAX nhìn thấy cả 8 chip cùng lúc; DreamerV3
mặc định chỉ dùng 1 chip cho train (`jax.train_devices: [0]`). Cách đơn giản và
đáng tin cậy nhất để tận dụng 8 chip trên một model nhỏ (`size1m`) là chạy nhiều
seed **song song trong background**, mỗi seed ăn 1 chip qua biến môi trường XLA:

```bash
%%bash
cd /kaggle/working/calculus-of-learning/third_party/dreamerv3
TASK=dmc_walker_walk
for SEED in 0 1 2 3 4; do
  CHIP=$((SEED % 8))
  ( TPU_VISIBLE_CHIPS=$CHIP python dreamerv3/main.py \
      --logdir ../../outputs/dreamer/${TASK}/seed${SEED} \
      --configs dmc_proprio --task ${TASK} \
      --seed ${SEED} --run.steps 5e5 --jax.platform tpu \
      > ../../outputs/dreamer/${TASK}_seed${SEED}.log 2>&1 & )
done
wait
```

> `TPU_VISIBLE_CHIPS` pins một process vào một chip cụ thể trên TPU v5e-8. Nếu
> biến này không có hiệu lực trên image Kaggle hiện tại (kiểm bằng
> `jax.devices()` bên trong mỗi process — chỉ nên thấy 1 device), chạy tuần tự
> thay vì song song (bỏ `&`/`wait`, dùng vòng `for` thường) — chậm hơn nhưng an
> toàn.

**GPU T4 x2:** 2 GPU riêng biệt, pin bằng `CUDA_VISIBLE_DEVICES` — cách chắc ăn
nhất, không phụ thuộc cờ nội bộ:

```bash
%%bash
cd /kaggle/working/calculus-of-learning/third_party/dreamerv3
TASK=dmc_walker_walk
for SEED in 0 1 2 3 4; do
  GPU=$((SEED % 2))
  ( CUDA_VISIBLE_DEVICES=$GPU python dreamerv3/main.py \
      --logdir ../../outputs/dreamer/${TASK}/seed${SEED} \
      --configs dmc_proprio --task ${TASK} \
      --seed ${SEED} --run.steps 5e5 --jax.platform cuda \
      > ../../outputs/dreamer/${TASK}_seed${SEED}.log 2>&1 & )
  # cap 2 runs đồng thời — chờ trước khi thả seed tiếp theo nếu SEED lẻ
  [ $((SEED % 2)) -eq 1 ] && wait
done
wait
```

**GPU RTX 6000 (1 GPU, VRAM lớn):** không có device để song song — chạy tuần tự,
tận dụng VRAM bằng cách tăng `batch_size` thay vì chia seed:

```bash
%%bash
cd /kaggle/working/calculus-of-learning/third_party/dreamerv3
TASK=dmc_walker_walk
for SEED in 0 1 2 3 4; do
  python dreamerv3/main.py \
    --logdir ../../outputs/dreamer/${TASK}/seed${SEED} \
    --configs dmc_proprio --task ${TASK} \
    --seed ${SEED} --run.steps 5e5 --batch_size 32 --jax.platform cuda
done
```

Chạy lại cell sau khi tiến trình bị ngắt giữa chừng (trong cùng phiên) là tiếp
tục từ checkpoint gần nhất trong `logdir`.

> Giai đoạn B–D (probe rollout `E(h)`, distill block predictor, analyze) — code
> theo spec §7 trong `experiments/e1_error_laws/dreamer/`; **chưa implement**,
> sẽ bổ sung. Train xong giai đoạn A trước là đúng thứ tự.

## Cell 5 — E2 Frame Freedom vs OOD (~1–2h toàn bộ, chạy tốt trên mọi accelerator)

```bash
!bash scripts/run_e2_frame_freedom.sh --env walker-walk
!bash scripts/run_e2_frame_freedom.sh --env cheetah-run
```

Đọc kỹ output bước `[2/5]` (shift-strength oracle) — nếu có `[WARN]` thì dừng, xem
spec `docs/E2-Frame-Freedom-vs-OOD-Generalization.md` §6. Với T4x2/TPU v5e-8, các
run độc lập của E2 (5 config × 3 seed) cũng có thể song song hoá bằng đúng pattern
ở Cell 4b nếu cần rút ngắn thời gian — mặc định script chạy tuần tự vì mỗi run đã
rất nhẹ (vài phút).

## Cell 6 — Đóng gói kết quả trong phiên

```bash
!zip -r results.zip outputs/results outputs/plots
```
Tải qua File Browser (sidebar trái, `/kaggle/working/results.zip`).

## Cell 7 — Giữ checkpoint qua nhiều phiên (Save Version)

Trước khi phiên hết giờ hoặc bị đóng: **Save Version** (góc trên phải) →
"Save & Run All" hoặc "Quick Save" — mọi thứ trong `/kaggle/working` lúc đó trở
thành **Output** của version, tải lại được ở phiên sau qua tab Output, hoặc add
làm input dataset cho notebook kế tiếp.

## Cell 7b — (Tuỳ chọn) Đẩy checkpoint lên Kaggle Dataset để resume nhanh hơn

Nếu sweep dài hơn 1 phiên: nén `outputs/dreamer` thành dataset riêng, phiên sau
attach dataset đó làm input và symlink vào `outputs/dreamer` trước khi chạy tiếp
— tránh phải tải lại Output cả notebook mỗi lần.

```bash
!kaggle datasets version -p outputs/dreamer -m "checkpoint update" \
  || kaggle datasets init -p outputs/dreamer   # lần đầu: tạo dataset mới rồi 'version'
```
Cần `kaggle.json` API token (Kaggle Account → Create New Token) đặt ở
`~/.kaggle/kaggle.json` — thường không cần trong chính notebook Kaggle (đã có sẵn
quyền ghi dataset qua `kaggle` CLI được cài kèm).

---

## Tóm tắt thời gian (ước tính — đo lại bằng Cell 4a)

| Thí nghiệm | Thiết bị | Thời gian |
|---|---|---|
| E1 Tầng 0 (toy) | CPU/bất kỳ | ~30 phút |
| E1 Tầng 1, 1 run | TPU v5e-8 (1 chip) / T4 / RTX 6000 | ~1–2h |
| E1 Tầng 1, đủ 15 runs | TPU v5e-8, 8 chip song song | ~2 lượt song song ≈ 4h/task |
| E1 Tầng 1, đủ 15 runs | GPU T4 x2, song song 2 | ~2.5 lượt ≈ 3–5h/task |
| E1 Tầng 1, đủ 15 runs | GPU RTX 6000, tuần tự | ~5 lượt ≈ 5–10h/task |
| E2, 1 env (15 runs) | bất kỳ có JAX | ~1h |
