"""Rule catalogue: what each check means and how to fix it."""
from __future__ import annotations

from dataclasses import dataclass

SEVERITIES = ("high", "medium", "low")


@dataclass(frozen=True)
class Rule:
    id: str
    title: str
    fix: str


RULES = {r.id: r for r in [
    Rule("PX001", "Preprocessing fitted before the train/test split",
         "Split first. Fit the scaler, encoder, selector or resampler on the training part only, "
         "then apply it to the test part, or wrap both steps in a scikit-learn/imblearn Pipeline."),
    Rule("PX002", "Test set used to choose the model",
         "Hold out a separate validation set (or use cross-validation) for early stopping, "
         "checkpoint choice and tuning. Touch the test set once, at the end."),
    Rule("PX003", "Model scored on the data it was trained on",
         "Score the model on rows it never saw during fit: a held-out test set or cross_val_score."),
    Rule("PX004", "Randomness without a fixed seed",
         "Pass random_state=<int> to splitters, and call torch.manual_seed / np.random.seed / "
         "random.seed once at start-up, so reruns give the same numbers."),
    Rule("PX005", "Evaluation without model.eval()",
         "Call model.eval() before evaluating (and model.train() before training again); "
         "torch.no_grad() only turns off gradients, not dropout or batch-norm updates."),
    Rule("PX006", "Fitting on test data",
         "Fit only on training data. For test data call transform() / predict() with the "
         "already-fitted object."),
    Rule("PX007", "Training continues in eval mode",
         "Call model.train() at the start of every training epoch, because the evaluation "
         "step switched the model to eval mode."),
]}
