# Feature classifier trainer

Run from `feature_extractors`:

```bash
python -m trainer.train --backbone base --modalities all --seeds 42 52 62 --device cuda
python -m trainer.train --backbone large --modalities T+A T+A+V --seeds 42 52 62 --device cuda
```

For Chinese MCSD1, add `--dataset mcsd1`; the defaults then point to
`features/mcsd1` and `outputs/mcsd1`. Inputs are
`features/{dataset}/{extractor}/{train,valid,test}.npz`. The loader
matches rows by the archive's `keys` (MCSD1 `File Name`), verifies labels, feature dimensions, and split
isolation, then pools the eight video frames to one vector. Mean and standard
deviation are fitted on train only. Validation chooses the best checkpoint by
macro-F1; ties prefer lower validation loss. Test is evaluated once per seed
after loading that checkpoint.

Defaults: AdamW, 10% linear warm-up followed by cosine decay, gradient clipping
at 1.0, up to 50 epochs, at least 5 epochs, patience 8, and a 256-d shared
space. `--amp` enables CUDA mixed precision. `--class-weight balanced` and
`--label-smoothing` are optional. No checkpoint or early stopping decision
uses the test split.

Each seed writes `config.json`, `train_normalization.npz`, `best.pt`,
`history.json`, `result.json`, and validation/test prediction CSVs under
`outputs/{dataset}/{backbone}/{modality}/seed_{seed}/`. Each modality directory
also contains `summary.json` with the test mean and sample standard deviation
over seeds. Existing run directories require `--overwrite`.
