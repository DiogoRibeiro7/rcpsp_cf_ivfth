# Usage Guide

This guide explains how to tailor the bundled toy instance and configure IVF-TH targets for your project data.

## 1. Edit `build_toy_instance`

The reference dataset lives in `rcpsp_cf_ivfth/examples/toy_instance.py`. The `build_toy_instance` function returns three objects:

```python
activities, finance, calendar = build_toy_instance()
```

- `activities`: `Dict[str, Activity]`
  - Each `Activity` defines a name, predecessor list, and a dictionary of `ModeData`.
  - `ModeData` stores a fuzzy `duration` and resource consumption as `NIVTF` numbers, plus a `payment` value.
- `finance`: `FinanceParams`
  - Captures the economic environment (interest rates, credit limits, initial cash, resource unit costs, etc.).
- `calendar`: `CalendarParams`
  - Defines the time horizon (`T_days`) and accounting periods (`Y_periods`) used for cash aggregation.

### Steps to customize

1. **Duplicate the template**: Copy `build_toy_instance` and rename it (e.g., `build_case_study_instance`) to keep the toy data intact.
2. **List activities**: Replace the sample entries with your activity identifiers, predecessors, and feasible modes.
3. **Encode fuzzy values**:
   - Create NIVTF numbers with `NIVTF(*create_triangle(lower, modal, upper))` or by specifying all six parameters directly.
   - Ensure ordering respects the constraints described in the docstring (`ao_U < ao_L < am_L == am_U < ap_L < ap_U`).
4. **Update resource demands**: Populate `renewables` (per-day usage) and `nonrenewables` (per-activity consumption) using NIVTF values.
5. **Set payments**: Record cash inflows (positive) or outflows (negative) at activity completion via the `payment` field.
6. **Adjust finance parameters**: Tune interest rates, credit caps, initial capital, and minimum cash floors in the `FinanceParams` block.
7. **Define the calendar**: Set `T_days` to the scheduling horizon and update `Y_periods` so each tuple `(start, end)` covers contiguous day ranges.

After editing, you can export the new factory function from `rcpsp_cf_ivfth/examples/__init__.py` to make it importable elsewhere.

## 2. Configure IVF-TH targets

The IVF-TH scalarization relies on aspiration (`PiS`) and nadir (`NiS`) targets for each objective, as defined by the `IVFTHTargets` dataclass:

```python
targets = IVFTHTargets(
    alpha_level=0.5,
    Z1_PIS=40.0,
    Z1_NIS=120.0,
    Z2_PIS=25000.0,
    Z2_NIS=0.0
)
```

Follow these steps to set meaningful values:

1. **Choose an alpha level**: `alpha_level` controls the blend between the lower and upper bounds of your NIVTF numbers. Common choices are 0.5 (central) or a risk-aware value such as 0.7.
2. **Estimate `Z1_PIS` (best makespan)**:
   - Solve a single-objective model that minimizes makespan (set `theta1=1.0`, `theta2=0.0`) or use domain knowledge to identify a tight lower bound.
   - Record the best feasible makespan as PiS.
3. **Estimate `Z1_NIS` (worst makespan)**:
   - Use the scheduling horizon or a relaxed run (e.g., penalty weights flipped) to obtain a conservative upper bound.
4. **Estimate `Z2_PIS` (best final cash flow)**:
   - Run a max cash-flow optimization (`theta1=0.0`, `theta2=1.0`) or compute the highest achievable surplus from your financial plan.
5. **Estimate `Z2_NIS` (worst final cash flow)**:
   - Set a safe lower bound (commonly zero or the minimum allowable cash floor in `FinanceParams`).
6. **Set IVF-TH weights** (`IVFTHWeights`):
   - Adjust `theta1`, `theta2`, and `gamma_tradeoff` to trade off between individual objective memberships and the max-min fairness term.

Document the procedure you used to derive PiS/NiS values so future runs remain reproducible.

## 3. Run your scenario

```bash
python -m rcpsp_cf_ivfth.examples.toy_instance \
  --solver cbc \
  --alpha 0.5 \
  --pis 40 25000 \
  --nis 120 0
```

If you expose multiple scenario builders, wire them to CLI arguments or environment variables so you can switch datasets without editing package code.

## 4. Validate results

- Run `pytest` to ensure your custom instance still satisfies the structural tests in `tests/test_examples.py`.
- Enable solver logs (`solver_options={"tee": True}`) when debugging infeasibilities or non-optimal status reports.
