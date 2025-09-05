# Signals-And-Systems - Programming HW

A hands-on repo for four signal-processing problems (frequency translation & mixing, DFT/IDFT from scratch, channel equalization, and speech de-noising).
All four problems are implemented as Jupyter notebooks, with ready-to-listen WAV outputs.

> **Problems PDF**: the assignment text is in **`Signal_PHW_questions.pdf`** at the repo root.

---

## What’s inside

* **Problem 1 — Frequency translation & selective mixing**
  Shift one audio to ultrasonic, mix with a second, then shift back + low-pass to reveal either source. Robust audio I/O, Hilbert-based frequency shifter, spectra plots, and exports:

  * `part_a_music1_shifted.wav`
  * `part_b_mix.wav`
  * `part_c_mix_shifted_back.wav`
  * `part_c_recovered_m1_lowpass.wav`

* **Problem 2 — Discrete Fourier Transform (DFT)**
  Derive IDFT, implement `fft`/`ifft` in pure NumPy (O(N²), **not** the fast FFT), verify against `numpy.fft`, and explore sampling of $x(t)=\cos(\pi t)$ with the effects of **N** and **T** on the DFT.

* **Problem 3 — Channel equalization**
  Treat an unknown system as LTI. Estimate the channel with an **STFT least-squares ratio** using `clean1.wav`/`distorted1.wav`, design a **regularized inverse** with a gain cap, and recover `distorted2.wav`:

  * `recovered2.wav`
  * `recovered2_without_normalization.wav`

* **Problem 4 — Signal de-noising**
  STFT mask with noise tracking (tail bootstrap + minimum statistics), SNR-adaptive **oversubtraction**, temporal/frequency smoothing, and alternatives (notch/HPF, decision-directed Wiener, Log-MMSE). Outputs (under `output/denoise_outputs/`):

  * `denoise1.wav` (moderate)
  * `denoise1_all_aggressive.wav`
  * `denoise1_listen.wav` & `denoise1_all_aggressive_listen.wav` (peak-normalized for quick A/B)
  * Baselines: `denoise1_notch_only.wav`, `denoise1_ddwiener.wav`, `denoise1_logmmse.wav`
  * Metrics: `denoise_metrics.csv`

---

## Repository layout

```
.
├─ README.md
├─ Signal_PHW_Merged_4_implementations.ipynb   # One notebook runs all four problems end-to-end
├─ Signal_PHW_questions.pdf                    # The assignment PDF (all problems)
├─ input/                                      # Provided audio
│  ├─ clean1.wav
│  ├─ distorted1.wav
│  ├─ distorted2.wav
│  ├─ music1.wav
│  ├─ music2.wav
│  └─ noisy1.wav
├─ output/                                     # Collected outputs
│  ├─ denoise_outputs/
│  │  ├─ denoise1.wav
│  │  ├─ denoise1_all_aggressive.wav
│  │  ├─ denoise1_all_aggressive_listen.wav
│  │  ├─ denoise1_ddwiener.wav
│  │  ├─ denoise1_listen.wav
│  │  ├─ denoise1_logmmse.wav
│  │  ├─ denoise1_notch_only.wav
│  │  └─ denoise_metrics.csv
│  ├─ part_a_music1_shifted.wav
│  ├─ part_b_mix.wav
│  ├─ part_c_mix_shifted_back.wav
│  └─ part_c_recovered_m1_lowpass.wav
└─ Separated 4 problems/
   ├─ Problem 1/
   │  ├─ Signal_PHW_Q1.ipynb
   │  ├─ music1.wav
   │  ├─ music2.wav
   │  └─ output/
   │     ├─ part_a_music1_shifted.wav
   │     ├─ part_b_mix.wav
   │     ├─ part_c_mix_shifted_back.wav
   │     └─ part_c_recovered_m1_lowpass.wav
   ├─ Problem 2/
   │  └─ Signal_PHW_Q2.ipynb
   ├─ Problem 3/
   │  ├─ Signal_PHW_Q3.ipynb
   │  ├─ clean1.wav
   │  ├─ distorted1.wav
   │  ├─ distorted2.wav
   │  └─ Output/
   │     ├─ recovered2.wav
   │     └─ recovered2_without_normalization.wav
   └─ Problem 4/
      ├─ Signal_PHW_Q4.ipynb
      ├─ noisy1.wav
      ├─ denoise1.wav
      └─ denoise_outputs/  (same set as in /output/denoise_outputs/)
```

> You can either run the **merged notebook** at the root or the **per-problem notebooks** inside `Separated 4 problems/`.

---

## Requirements

* Python 3.9+
* Jupyter (Lab/Notebook) or any environment able to run `.ipynb`
* Libraries

  * `numpy`, `scipy`, `matplotlib`, `pandas`
  * `soundfile` (preferred for I/O; needs system lib **libsndfile**)
  * (Notebooks also use small pieces of `IPython.display` for inline audio)

**Quick install**

```bash
# optionally create/activate a venv first
pip install numpy scipy matplotlib pandas soundfile jupyter
# on Debian/Ubuntu you may need libsndfile:
# sudo apt-get install libsndfile1
```

---

## How to run

### Option A — one-shot (runs everything)

1. Open `Signal_PHW_Merged_4_implementations.ipynb`.
2. Run all cells.
   Outputs are written under `output/` (and `output/denoise_outputs/`).

### Option B — per problem

* **Problem 1:** open `Separated 4 problems/Problem 1/Signal_PHW_Q1.ipynb`
  Input: `music1.wav`, `music2.wav` → Outputs in the local `output/` subfolder.
* **Problem 2:** open `Separated 4 problems/Problem 2/Signal_PHW_Q2.ipynb`
* **Problem 3:** open `Separated 4 problems/Problem 3/Signal_PHW_Q3.ipynb`
  Inputs: `clean1.wav`, `distorted1.wav`, `distorted2.wav` → Outputs under `Output/`.
* **Problem 4:** open `Separated 4 problems/Problem 4/Signal_PHW_Q4.ipynb`
  Input: `noisy1.wav` → Outputs under `denoise_outputs/`.

> All notebooks are self-contained: they handle mono down-mix, resampling when needed, plotting, and file I/O.

---

## Problem-by-problem notes

### Problem 1 — Frequency translation & selective mixing

**Core idea:** make one source inaudible by shifting it far above the audible band, sum with the other, then **undo the shift** and low-pass to reveal the first source again.

* **Frequency shift:** analytic (Hilbert) method
  $x_a(t)\,e^{j2\pi f_{\text{shift}}t}$ → real part.
  Default `SHIFT_HZ = 18.5 kHz`. Keep $f_\text{max} + f_\text{shift} < \tfrac{sr}{2}$ to avoid aliasing.
* **Files produced**

  * `part_a_music1_shifted.wav`: almost inaudible alone (energy translated near 18.5 kHz).
  * `part_b_mix.wav`: mixture ≈ `music2` (shifted `music1` is out of band).
  * `part_c_mix_shifted_back.wav`: now `music1` reappears; `music2` is upshifted.
  * `part_c_recovered_m1_lowpass.wav`: optional low-pass to clean residual `music2` replica.

### Problem 2 — DFT/IDFT from first principles

* **Definitions**

  * DFT: $X[k]=\sum_{n=0}^{N-1} x[n]e^{-j2\pi kn/N}$
  * IDFT: $x[n]=\frac{1}{N}\sum_{k=0}^{N-1} X[k]e^{+j2\pi kn/N}$
* **Implementations**

  * `fft(x)` and `ifft(X)` as dense $N\times N$ matrix multiplies (**O(N²)**), for learning/verification.
* **Experiments**

  * Verify `ifft(fft(x)) ≈ x` (machine precision).
  * Sample $x(t)=\cos(\pi t)$ with $T=\tfrac12, N=1000$; show DFT spikes at $\omega=±\pi/2$ (i.e., $±0.5$ Hz).
  * Show impact of changing **N** (resolution/leakage) and **T** (Nyquist/aliasing).

### Problem 3 — Channel equalization

* **Estimate** $H(f)$ from the aligned pair (`clean1`, `distorted1`) via an **STFT least-squares ratio**
  $H(f)=\frac{\sum_t Y(f,t)X^*(f,t)}{\sum_t |X(f,t)|^2+\varepsilon}$
* **Inverse EQ** $G(f)=\frac{H^*(f)}{|H(f)|^2+\lambda}$ with **+18 dB gain cap**
* **Apply** $G(f)$ to `distorted2` → `recovered2.wav` (+ a raw, unnormalized version)
* **Plots**: magnitude/phase of $H$, before/after spectra & spectrograms, ratio $20\log_{10}|Rec2|/|Dist2|$ vs $20\log_{10}|G|$.

### Problem 4 — Speech de-noising (STFT mask)

* **Noise PSD**: seed from a **noise-only tail** (≈ 9.5 s → end) + **minimum statistics**
* **Mask**: SNR-adaptive **oversubtraction** with amplitude floor and time/frequency smoothing
* **Variants**: moderate vs aggressive; baselines (notch/HPF, DD-Wiener, Log-MMSE)
* **Outputs** (in `output/denoise_outputs/`)

  * `denoise1.wav` (moderate), `denoise1_all_aggressive.wav` (aggressive)
  * Listening-normalized versions and baselines
  * `denoise_metrics.csv` with objective results; typical values from the provided input:

| Metric                                | Moderate | Aggressive | Original |
| ------------------------------------- | -------: | ---------: | -------: |
| Tail RMS improvement (dB)             |   +19.78 |     +19.83 |        — |
| Attenuation on low-energy frames (dB) |   −20.69 |     −21.07 |        — |
| Mean gain on high-energy frames (dB)  |    −6.86 |      −8.51 |        — |
| Spectral flatness (median)            |    0.106 |      0.077 |    0.273 |

*(Exact numbers may vary slightly across platforms.)*

---

## Repro tips & knobs

* **Audio I/O**

  * `soundfile` is preferred; if missing, notebooks fall back to `scipy.io.wavfile`.
  * All signals are converted to mono `float32` in $[-1,1]$. Writers clip/scale safely.
* **Sample-rate mismatches** are handled with rational resampling (`scipy.signal.resample_poly`).
* **Problem 1:** If your `music1` has energy above \~3–4 kHz or your `sr` is not 44.1 kHz, reduce `SHIFT_HZ` to keep $f_\text{max}+SHIFT_HZ < sr/2$.
* **Problem 3:** The inverse EQ uses Tikhonov regularization (`λ`) and a **gain cap** to avoid noise blow-up in channel notches. Increase the cap for more aggressive recovery (riskier).
* **Problem 4:** Main quality–noise trade-offs:

  * `ALPHA_TAIL` (oversubtraction upper bound), `BETA_FLOOR` (amplitude floor),
  * temporal smoothing `SMOOTH_T` and frequency smoothing `SMOOTH_F_BINS`,
  * FFT size/HOP (resolution vs. smearing).

---

## FAQ

**Q: The audio widgets don’t play.**
A: Ensure you’re in a Jupyter environment with audio enabled. You can always open the WAVs from the `output/` folders in any player.

**Q: I don’t have `libsndfile`.**
A: Install it (Linux: `apt-get install libsndfile1`). Otherwise SciPy’s WAV writer is used (int16).

**Q: Why are there both normalized and non-normalized files?**
A: “*listen*” files are peak-normalized for quick A/B comparisons in notebooks. The non-normalized files keep natural levels for analysis/metrics.

---

## Acknowledgements

* The assignment text is in **`Signal_PHW_questions.pdf`**.
* Implementations rely on NumPy, SciPy, Matplotlib, and SoundFile.

---

## License

This repository is released under the **MIT License**. See `LICENSE` if present; otherwise you may reuse the code with attribution.

---

## Author

Sina Beyrami — Signals & Systems Practical HW.
