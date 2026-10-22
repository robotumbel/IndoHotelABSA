# E9 — Zero-shot LLM not used for drafting, on test-400 and blind-100

Encoders shown with seed-42 predictions (5-seed means come with E5).

| System | Joint-F1 test-400 | ACD-mF1 test-400 | blind-100 vs gold | blind-100 vs blind | Δ | ACD-mF1 vs blind |
|---|---:|---:|---:|---:|---:|---:|
| Gemini draft (silver) | 0.872 | 0.933 | 0.883 | 0.757 | -0.126 | 0.848 |
| IndoBERT (seed 42) | 0.761 | 0.862 | 0.767 | 0.732 | -0.035 | 0.838 |
| HI-ABSA (seed 42) | 0.759 | 0.867 | 0.776 | 0.734 | -0.041 | 0.841 |
| qwen25_7b_zeroshot (strict parser) | 0.028 | 0.028 | 0.026 | 0.026 | +0.000 | 0.024 |
| qwen25_7b_zeroshot (lenient parser) | 0.686 | 0.764 | 0.696 | 0.678 | -0.018 | 0.763 |
| qwen25_14b_zeroshot (strict parser) | 0.353 | 0.410 | 0.335 | 0.311 | -0.024 | 0.343 |
| qwen25_14b_zeroshot (lenient parser) | 0.721 | 0.816 | 0.745 | 0.721 | -0.024 | 0.798 |

Format compliance of the open-LLM outputs:

- qwen25_7b_zeroshot: 400 reviews; output begins with a JSON array in 13; strict parser non-empty 6; lenient parser non-empty 374.
- qwen25_14b_zeroshot: 400 reviews; output begins with a JSON array in 82; strict parser non-empty 123; lenient parser non-empty 384.

Review-level bootstrap on blind-100 (2,000 resamples), Joint-F1 gap:

| Comparison | vs adjudicated gold | vs blind majority |
|---|---|---|
| qwen25_7b_zeroshot (lenient parser) − Gemini draft (silver) | -0.188 [-0.242, -0.132] | -0.079 [-0.127, -0.032] |
| qwen25_7b_zeroshot (lenient parser) − IndoBERT (seed 42) | -0.071 [-0.125, -0.020] | -0.055 [-0.104, -0.005] |
| qwen25_14b_zeroshot (lenient parser) − Gemini draft (silver) | -0.138 [-0.184, -0.092] | -0.036 [-0.072, -0.002] |
| qwen25_14b_zeroshot (lenient parser) − IndoBERT (seed 42) | -0.022 [-0.058, +0.014] | -0.012 [-0.049, +0.024] |
