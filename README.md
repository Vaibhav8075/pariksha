# pariksha

**pariksha** (परीक्षा, "exam") checks machine-learning code for the mistakes that make a
model look better on paper than it really is: test data leaking into training, the test set
being used to pick the model, scoring on training data, and evaluation that runs in the wrong
mode. It reads `.py` files and Jupyter notebooks, needs no extra packages, and points to the
exact file and line (or notebook cell) for every finding.

With `--explain`, each finding is sent to the [BharatCode](https://bharatcode.ai) API, which
explains it in plain words and suggests the corrected code.

Built for the BharatCode Build League 2026.

## Why

A model can score 99% in a notebook and still fail on real data, because the number was
measured wrongly. The usual causes are small and easy to miss in review: a `StandardScaler`
fitted before `train_test_split`, SMOTE run on the full dataset, `validation_data=(X_test,
y_test)` with early stopping, or an evaluation loop without `model.eval()`. None of these
crash, and all of them inflate the reported score. pariksha looks for these patterns in the
code itself, before anyone trusts the result.

## Install

```bash
pip install git+https://github.com/Vaibhav8075/pariksha
```

Python 3.10 or newer. There are no runtime dependencies.

## Use

```bash
pariksha                       # scan the current folder
pariksha train.py notebooks/   # scan files and folders
pariksha --json > report.json  # machine-readable output
pariksha --list-rules          # what each rule means
```

`python -m pariksha` works too. Folders such as `.git`, `venv`, `node_modules` and
`.ipynb_checkpoints` are skipped.

Example, run on [`examples/leaky_project`](examples/leaky_project):

```text
examples/leaky_project/churn.ipynb [cell 3]:4  MEDIUM  PX001  Preprocessing fitted before the train/test split
    SimpleImputer is fitted on `X` before train_test_split (cell 4 line 1), so it has already seen the rows that end up in the test set.
    > X = SimpleImputer(strategy='mean').fit_transform(X)
    fix: Split first. Fit the scaler, encoder, selector or resampler on the training part only, then apply it to the test part, or wrap both steps in a scikit-learn/imblearn Pipeline.

examples/leaky_project/train_keras.py:13  HIGH    PX006  Fitting on test data
    MinMaxScaler is fitted on `X_test`, so the test data is scaled with its own statistics instead of the ones learned from training data.
    > X_test = MinMaxScaler().fit_transform(X_test)
    fix: Fit only on training data. For test data call transform() / predict() with the already-fitted object.

...
4 files scanned, 13 findings (7 high, 4 medium, 2 low)
```

The same programs written correctly, in [`examples/clean_project`](examples/clean_project),
give `0 findings`.

## Rules

| Rule | Finds | Severity |
|---|---|---|
| PX001 | A scaler, imputer, encoder, PCA, vectorizer, feature selector or resampler (SMOTE etc.) fitted on data that is later passed to `train_test_split`, followed through intermediate variables | high for label-aware selectors and resamplers, medium for the rest |
| PX002 | The test set used to steer training: `validation_data`, `eval_set`, `valid_sets` or `eval_dataset` given test data, a scheduler stepped on a test metric, or the "best" epoch chosen by a test score | high when early stopping, checkpoints or saving are involved |
| PX003 | A model scored (`score`, or a metric on `predict`) on exactly the data passed to its `fit` | high |
| PX004 | `train_test_split` or shuffling CV splitters without `random_state`; PyTorch training with no seed anywhere in the project | low |
| PX005 | An evaluate/test/predict function that runs under `torch.no_grad()` but never calls `model.eval()` | medium |
| PX006 | `fit`, `fit_transform` or `fit_resample` called on test data | high |
| PX007 | A training loop that switches to eval mode for validation and never switches back with `model.train()` | medium |

The exit status is 1 when a finding at or above `--fail-on` (default `high`) exists, so it
can run in CI or a pre-commit hook. `--fail-on none` always exits 0.

### Silencing a finding

Add a comment on the reported line:

```python
X = scaler.fit_transform(X)  # pariksha: ignore[PX001]
model.fit(X_test, y_test)    # pariksha: ignore
```

## Explanations with BharatCode

```bash
cp .env.example .env    # then put your key in BHARATCODE_API_KEY
pariksha --explain
```

For each finding (highest severity first, five by default, `--max-explain N` to change),
pariksha calls BharatCode's OpenAI-compatible chat API with the default model `qwen-3.8-27b`
and prints a short explanation plus a corrected version of the marked lines under the finding.
`BHARATCODE_MODEL` and `BHARATCODE_BASE_URL` override the model and endpoint. The model's
step-by-step reasoning mode is switched off for these calls: with it on, Qwen used its whole
token budget on reasoning and returned no answer, while with it off each explanation took a
few seconds.

A real response, for the `MinMaxScaler().fit_transform(X_test)` line in the Keras example:

```text
examples/leaky_project/train_keras.py:13  HIGH    PX006  Fitting on test data
    MinMaxScaler is fitted on `X_test`, so the test data is scaled with its own statistics instead of the ones learned from training data.
    > X_test = MinMaxScaler().fit_transform(X_test)
    fix: Fit only on training data. For test data call transform() / predict() with the already-fitted object.
    explanation (BharatCode):
      Fitting a new scaler on the test set leaks information about the test data's distribution into the preprocessing step. This means the test data is scaled using its own min/max values rather than the training set's, which is not how the model would be deployed in production. Consequently, the reported test accuracy is artificially inflated and does not reflect true generalization performance. You should reuse the scaler already fitted on the training data to transform the test data.

      ```python
      X_test = scaler.transform(X_test)
      ```
```

The explanation is model output, so read it as advice: in this case "artificially inflated" is
not guaranteed, since a test set scaled on its own can also score lower.

What is sent: the rule, the finding message and the 13 lines around the finding. Nothing else
from the project leaves the machine, and without `--explain` no network call is made. Without
a key, `--explain` prints a warning and shows the findings as usual.

## How it works

Every file is parsed with Python's `ast` module; nothing is imported or executed. Notebooks
are joined cell by cell, with IPython magics (`%matplotlib`, `!pip`) blanked and cells that
still do not parse skipped, so line numbers map back to the right cell.

PX001 follows assignments backwards from each `train_test_split` call to find every
variable its inputs were computed from, and flags fit calls on those variables that happen
before the split. Test data is recognised by name (`X_test`, `test_loader`, `testset`,
`yTest`), which is how most code names it.

## Limitations

- It is static analysis. Data passed between functions, stored in containers or loaded from
  files in between is not followed, so a leak spread across several modules can be missed.
- Test data is found by variable names. A validation set called `test_loader` is reported
  (rename it), and a test set called `holdout` is not.
- Pipelines (`make_pipeline`, `ColumnTransformer`) and cross-validation done inside them are
  treated as correct, which they are when used as intended.
- PX005 cannot see `model.eval()` called by the caller of a plain function; the message says
  so, and the finding can be silenced.

## Development

```bash
pip install -e ".[dev]"
python -m pytest
```

## Background

The rules cover leakage types described by Kapoor and Narayanan, who surveyed leakage across
published ML-based science, and the pitfalls listed in the scikit-learn user guide:

- S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based
  science," *Patterns*, vol. 4, no. 9, 2023.
- scikit-learn developers, "Common pitfalls and recommended practices,"
  https://scikit-learn.org/stable/common_pitfalls.html

## License

MIT, see [LICENSE](LICENSE).
