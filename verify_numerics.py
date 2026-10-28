"""Check the actual supplied numerical functions without importing experiment mains.

Usage: python verify_numerics.py [--src PATH] [--original-src PATH]
Only selected function ASTs and literal constants are executed.  Outputs are
verification_numerics.json and a concise stdout report, beside this script.
Finite differences use a fourth-order, symmetric five-point formula.  Clipped
tests deliberately avoid the nondifferentiable clipping threshold itself.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
FILES = {0: "n0", 0.5: "no.5", 1: "n1", 1.5: "n1.5", 2: "n2", 3: "n3", 4: "n4", 5: "n5"}
BASES = ("bernoulli", "chebyshev", "legendre")
BASIS_FUNCTIONS = {
    "bernoulli_basis", "chebyshev_basis_exact", "legendre_basis_exact",
    "_get_basis_raw", "get_column_norms", "get_basis", "forward", "uniform_nodes",
    "optimal_collocation",
}


def extract(path, names):
    """Load only requested definitions plus harmless literal module constants."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"), filename=str(path))
    ns = {"np": np, "comb": math.comb,
          "leggauss": np.polynomial.legendre.leggauss, "_norm_cache": {}}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                val = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    ns[target.id] = val
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            try:
                ns[node.target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                pass
    defs = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in names]
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(path), "exec"), ns)
    return ns


def load_base(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name.startswith(("loss_and_grad_", "compute_ode_loss_"))}
    ns = extract(path, BASIS_FUNCTIONS | names)
    ns["loss_name"] = next(n for n in names if n.startswith("loss_and_grad_"))
    ns["ode_name"] = next(n for n in names if n.startswith("compute_ode_loss_"))
    return ns


def finite_difference(fn, w, step=2e-5):
    w = np.asarray(w, dtype=float)
    initial = np.asarray(fn(w))
    out = np.empty(initial.shape + (w.size,))
    for j in range(w.size):
        d = np.zeros_like(w)
        h = step * max(1.0, abs(float(w.flat[j])))
        d.flat[j] = h
        out[..., j] = (-np.asarray(fn(w + 2*d)) + 8*np.asarray(fn(w+d))
                       - 8*np.asarray(fn(w-d)) + np.asarray(fn(w-2*d))) / (12*h)
    return out


def agreement(actual, expected, tolerance=2e-9):
    actual = np.asarray(actual).ravel()
    expected = np.asarray(expected).ravel()
    delta = np.abs(actual-expected)
    relative = float(np.linalg.norm(actual-expected) / max(1.0, np.linalg.norm(expected)))
    return {"max_absolute_error": float(delta.max(initial=0)),
            "relative_l2_scaled_by_max_1": relative, "tolerance": tolerance,
            "pass": relative < tolerance and bool(np.all(np.isfinite(actual)))}


def gradient_case(ns, basis, x, w, xmax):
    p, dp, d2p = ns["get_basis"](basis, x, w.size-1, xmax)
    loss, grad = ns[ns["loss_name"]](x, w, p, dp, d2p)
    fn = lambda weights: ns[ns["ode_name"]](x, weights, p, dp, d2p)
    numerical = finite_difference(fn, w)
    result = agreement(grad, numerical)
    result.update(loss=float(loss), independent_loss=float(fn(w)),
                  min_y=float(ns["forward"](x, w, p, dp, d2p)[0].min()))
    result["loss_agrees"] = bool(np.isclose(loss, fn(w), rtol=1e-12, atol=1e-13))
    result["pass"] &= result["loss_agrees"]
    return result


def derivative_cases(ns):
    x = np.concatenate((np.linspace(-1, 1, 57), [-1+1e-9, 1-1e-9]))
    out = []
    for basis, function, cls in [
        ("chebyshev", "chebyshev_basis_exact", np.polynomial.Chebyshev),
        ("legendre", "legendre_basis_exact", np.polynomial.Legendre),
    ]:
        values = ns[function](x, 11)
        for derivative, actual in enumerate(values):
            expected = np.column_stack([cls.basis(k).deriv(derivative)(x) for k in range(12)])
            check = agreement(actual, expected, tolerance=5e-12)
            check.update(basis=basis, derivative=derivative, max_degree=11)
            out.append(check)
    return out


def grid_cases(ns):
    xmax = float(ns.get("X_MAX", 1.0))
    nv = int(ns.get("N_VAL", 105))
    half = xmax / (2*nv)
    validation = np.linspace(1e-5 + half, xmax-half/2, nv)
    evaluation = np.linspace(1e-5, xmax, 500)
    train = ns["uniform_nodes"](int(ns.get("N_TRAIN", 30)), xmax)
    distance = lambda a, b: float(np.min(np.abs(a[:, None]-b[None, :])))
    tv, ve, te = distance(train, validation), distance(validation, evaluation), distance(train, evaluation)
    return {"train_size": len(train), "validation_size": len(validation), "evaluation_size": len(evaluation),
            "train_validation_min_distance": tv, "validation_evaluation_min_distance": ve,
            "train_evaluation_min_distance": te,
            "train_validation_disjoint": tv > 1e-12,
            "validation_evaluation_disjoint": ve > 1e-12,
            "train_evaluation_endpoint_overlap_expected": te <= 1e-12,
            "pass": tv > 1e-12 and ve > 1e-12}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=HERE/"src")
    parser.add_argument("--original-src", type=Path, default=None,
                        help="optional: source folder of the pre-revision code, to document which checks it failed")
    args = parser.parse_args()
    rng = np.random.default_rng(20261015)
    versions = [("after_rerun", args.src)]
    if args.original_src is not None and args.original_src.exists():
        versions.insert(0, ("before_original", args.original_src))
    report = {"source": args.src.name, "original_source": None if args.original_src is None else args.original_src.name,
              "method": "Actual selected source ASTs; five-point finite difference, step 2e-5; no training run",
              "positive_domain_gradients": [], "basis_derivatives": [], "grids": [],
              "fractional_clipped_domain": [], "xi1_jacobian": []}
    for n, stem in FILES.items():
        ns = load_base(args.src/(stem+".py"))
        x = np.linspace(1e-5, 1, 30)
        for basis in BASES:
            w = rng.normal(0, 0.1, (6, 1))
            check = gradient_case(ns, basis, x, w, 1.0)
            check.update(n=n, basis=basis)
            report["positive_domain_gradients"].append(check)
        report["basis_derivatives"].append({"n": n, "checks": derivative_cases(ns)})
        report["grids"].append({"n": n, **grid_cases(ns)})

    for n in (0.5, 1.5):
        stem = FILES[n]
        for version, src in versions:
            ns = load_base(src/(stem+".py"))
            for basis in BASES:
                x = np.linspace(1.7, 2.4, 30)
                w = np.zeros((6, 1)); w[0, 0] = -0.8
                check = gradient_case(ns, basis, x, w, 2.5)
                check.update(n=n, basis=basis, version=version)
                report["fractional_clipped_domain"].append(check)

    for version, src in versions:
        ns = extract(src/"xi1_domain.py", {"trial", "residual", "jacobian"})
        bmod = load_base(args.src/"n0.py")
        for n in FILES:
            for state in ("positive", "clipped"):
                x = np.linspace(1e-5, 1, 30) if state == "positive" else np.linspace(1.7, 2.4, 30)
                w = rng.normal(0, 0.05, 6) if state == "positive" else np.array([-0.8, 0, 0, 0, 0, 0.])
                p, dp, d2p = bmod["get_basis"]("bernoulli", x, 5, 2.5)
                expected = finite_difference(lambda v: ns["residual"](v, x, p, dp, d2p, n), w)
                actual = ns["jacobian"](w, x, p, dp, d2p, n)
                check = agreement(actual, expected)
                check.update(n=n, version=version, state=state)
                report["xi1_jacobian"].append(check)

    relevant = (report["positive_domain_gradients"] + report["grids"] +
                [c for item in report["basis_derivatives"] for c in item["checks"]] +
                [c for c in report["fractional_clipped_domain"] if c["version"] == "after_rerun"] +
                [c for c in report["xi1_jacobian"] if c["version"] == "after_rerun"])
    report["all_current_checks_pass"] = all(c["pass"] for c in relevant)
    report["failed_current_checks"] = [c for c in relevant if not c["pass"]]
    report["before_fix_failures_expected"] = {
        section: sum(not c["pass"] for c in report[section] if c["version"] == "before_original")
        for section in ("fractional_clipped_domain", "xi1_jacobian")}
    dest = HERE/"verification_numerics.json"
    dest.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"all_current_checks_pass": report["all_current_checks_pass"],
                      "positive_gradient_cases": len(report["positive_domain_gradients"]),
                      "before_fix_failures_expected": report["before_fix_failures_expected"],
                      "failed_current_checks": report["failed_current_checks"], "report": dest.name}, indent=2))
    raise SystemExit(0 if report["all_current_checks_pass"] else 1)


if __name__ == "__main__":
    main()
