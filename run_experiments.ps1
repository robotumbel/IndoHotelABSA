# ============================================================
#  run_experiments.ps1 — Jalankan SEMUA eksperimen Paper 2
#  Data : train_silver.jsonl (2480) / val_gold.jsonl (100) / test_gold.jsonl (400)
#  Hasil: results_experiments\*.txt  (tinggal salin ke tabel paper)
#
#  Cara pakai (di PowerShell):
#    cd "D:\Code-PhD\Churn Prediction\paper2_sentiment"
#    .\run_experiments.ps1
#
#  Catatan: di CPU tiap model encoder bisa 1-3 jam. Biarkan berjalan.
#  Boleh dihentikan (Ctrl+C) dan dijalankan ulang — model yang sudah
#  punya file prediksi akan DILEWATI otomatis.
# ============================================================

$TRAIN = "train_silver.jsonl"
$VAL   = "val_gold.jsonl"
$TEST  = "test_gold.jsonl"
$EPOCHS = 3
$OUT   = "results_experiments"

New-Item -ItemType Directory -Force -Path $OUT | Out-Null

function Evaluate($pred, $name) {
  if (Test-Path $pred) {
    Write-Host ">> Evaluasi $name" -ForegroundColor Cyan
    python evaluate_absa.py --gold $TEST --pred $pred 2>&1 |
      Tee-Object -FilePath (Join-Path $OUT "results_$name.txt")
  }
}

function RunStep($name, $pred, $cmd) {
  Write-Host ""
  Write-Host ("=" * 60) -ForegroundColor Yellow
  if (Test-Path $pred) {
    Write-Host "[SKIP] $name -> $pred sudah ada" -ForegroundColor DarkGray
  } else {
    Write-Host "[RUN ] $name" -ForegroundColor Green
    Invoke-Expression $cmd 2>&1 | Tee-Object -FilePath (Join-Path $OUT "log_$name.txt")
  }
  Evaluate $pred $name
}

# ── 1. Lexicon baseline (instan) ─────────────────────────────
RunStep "lexicon" "pred_lex_gold.jsonl" `
  "python lexicon_baseline.py --test $TEST --pred_out pred_lex_gold.jsonl"

# ── 2. HI-ABSA (proposed) + ablasi ───────────────────────────
RunStep "hiabsa_full" "pred_hiabsa_full.jsonl" `
  "python finetune_hiabsa.py --train $TRAIN --val $VAL --test $TEST --epochs $EPOCHS --pred_out pred_hiabsa_full.jsonl"

RunStep "hiabsa_noaqa" "pred_hiabsa_noaqa.jsonl" `
  "python finetune_hiabsa.py --train $TRAIN --val $VAL --test $TEST --epochs $EPOCHS --no_aqa --pred_out pred_hiabsa_noaqa.jsonl"

RunStep "hiabsa_nolgf" "pred_hiabsa_nolgf.jsonl" `
  "python finetune_hiabsa.py --train $TRAIN --val $VAL --test $TEST --epochs $EPOCHS --no_lgf --pred_out pred_hiabsa_nolgf.jsonl"

RunStep "hiabsa_none" "pred_hiabsa_none.jsonl" `
  "python finetune_hiabsa.py --train $TRAIN --val $VAL --test $TEST --epochs $EPOCHS --no_aqa --no_lgf --pred_out pred_hiabsa_none.jsonl"

# ── 3. Encoder baselines ─────────────────────────────────────
RunStep "indobert" "pred_indobert_gold.jsonl" `
  "python finetune_indobert_absa.py --train $TRAIN --val $VAL --test $TEST --epochs $EPOCHS --model indobenchmark/indobert-base-p1 --pred_out pred_indobert_gold.jsonl"

RunStep "indobertweet" "pred_indobertweet.jsonl" `
  "python finetune_indobert_absa.py --train $TRAIN --val $VAL --test $TEST --epochs $EPOCHS --model indolem/indobertweet-base-uncased --pred_out pred_indobertweet.jsonl"

RunStep "mbert" "pred_mbert.jsonl" `
  "python finetune_indobert_absa.py --train $TRAIN --val $VAL --test $TEST --epochs $EPOCHS --model bert-base-multilingual-cased --pred_out pred_mbert.jsonl"

RunStep "xlmr" "pred_xlmr.jsonl" `
  "python finetune_indobert_absa.py --train $TRAIN --val $VAL --test $TEST --epochs $EPOCHS --model xlm-roberta-base --pred_out pred_xlmr.jsonl"

# ── 4. LLM LoRA (SFT generatif) ──────────────────────────────
RunStep "llm_lora" "pred_llm_lora_gold.jsonl" `
  "python finetune_llm_lora.py --train $TRAIN --val $VAL --test $TEST --epochs $EPOCHS --model Qwen/Qwen2.5-0.5B-Instruct --pred_out pred_llm_lora_gold.jsonl"

# ── 5. LLM few-shot (opsional; perlu model HF lokal) ─────────
# Buka komentar bila mau:
# RunStep "llm_fewshot" "pred_llm_fewshot.jsonl" `
#   "python llm_fewshot_absa.py --test $TEST --shots 3 --backend hf --model Qwen/Qwen2.5-0.5B-Instruct --pred_out pred_llm_fewshot.jsonl"

# ── Ringkasan ────────────────────────────────────────────────
Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Yellow
Write-Host "SELESAI. Ringkasan hasil (macro-F1 / joint):" -ForegroundColor Green
Get-ChildItem (Join-Path $OUT "results_*.txt") | ForEach-Object {
  Write-Host ""
  Write-Host $_.Name -ForegroundColor Cyan
  Get-Content $_.FullName | Select-String -Pattern "MACRO-F1|MICRO-F1|Akurasi|Joint|Precision"
}
Write-Host ""
Write-Host "Semua detail di folder: $OUT" -ForegroundColor Green
