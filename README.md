**How to reproduce:**
1) `make gk audit har garch`
2) `python src/build_sequences.py --symbol SPY --seq_len 120`
3) `python src/train_patchtst_multitask.py --symbol SPY --seq_len 120 --split_date 2023-01-02 --epochs 80 --batch 128 --patch_len 30 --d_model 128 --nhead 4 --nlayers 3 --beta 0.5`
4) `python src/eval_phase4.py --symbol SPY --holdout_start 2023-01-02`
