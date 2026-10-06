"""Draw the smallest exact cycle using only the standard library.

Run from any directory. --check compares the drawing to the committed SVG.
Exact arithmetic determines all labels and checks; floats are used only for
screen coordinates. This is a construction diagram, not a sampled experiment.
"""
import argparse
from fractions import Fraction
from html import escape
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from certify import backup, family, make_model, policy_value_check, selected

OUTPUT = ROOT / "docs" / "two-phase-cycle.svg"


def render():
    k, beta = 2, Fraction(1, 4)
    construction = family(k, beta)
    model = make_model(construction, 1)
    qa, qb = construction["qa"], construction["qb"]
    if backup([qa], model, beta, k)[0] != [qb]:
        raise RuntimeError("Q_A does not back up to Q_B")
    if backup([qb], model, beta, k)[0] != [qa]:
        raise RuntimeError("Q_B does not back up to Q_A")
    va, vb = policy_value_check(construction, model, beta, 1)
    optimal_qb = construction["mean_b"] + beta * va
    if va - optimal_qb != construction["c"] - construction["mean_b"]:
        raise RuntimeError("True action-value gap is inconsistent")

    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 690" '
        'role="img" aria-labelledby="title description">',
        '<title id="title">An exact two-phase quantile-control cycle</title>',
        '<desc id="description">One state, two actions, two quantiles, discount one quarter. '
        'The projected means select A in Q_A and B in Q_B. Exact Bellman backups '
        'swap these phases. True expected returns favor A in both phases.</desc>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" '
        'refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" '
        'fill="#334155"/></marker></defs>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#172033}'
        '.small{font-size:15px}.body{font-size:18px}.heading{font-size:23px;font-weight:700}'
        '.axis{stroke:#cbd5e1;stroke-width:1}.arrow{fill:none;stroke:#334155;'
        'stroke-width:2;marker-end:url(#arrow)}</style>',
        '<rect width="1000" height="690" fill="#fff"/>',
    ]

    def text(x, y, label, css="body", anchor="start"):
        elements.append(f'<text x="{x}" y="{y}" class="{css}" '
                        f'text-anchor="{anchor}">{escape(str(label))}</text>')

    text(32, 39, "Exact two-cycle; true values still favor A", "heading")
    text(32, 68, f"One environment state · K = {k} · discount = {beta}")
    reward_law = ", ".join(f"{reward} (p={prob})" for reward, prob in
                           zip(construction["rewards"], construction["weights"]))
    text(32, 96, f"Rewards: A = {construction['c']}; B = {reward_law}", "small")

    for left, phase, critic in ((32, "Q_A", qa), (526, "Q_B", qb)):
        greedy = selected(critic)[1]
        elements.append(f'<rect x="{left}" y="122" width="442" height="323" '
                        'rx="10" fill="#f8fafc" stroke="#cbd5e1"/>')
        text(left + 20, 158, f"{phase}: greedy action {'AB'[greedy]}", "heading")
        axis_left, axis_width = left + 67, 346

        def x_position(value):
            return axis_left + float((value - Fraction(1, 2)) / Fraction(3, 4)) * axis_width

        for tick in (Fraction(1, 2), Fraction(3, 4), Fraction(1), Fraction(5, 4)):
            x = x_position(tick)
            elements.append(f'<line x1="{x:.2f}" y1="180" x2="{x:.2f}" '
                            'y2="335" class="axis"/>')
            text(f"{x:.2f}", 362, tick, "small", "middle")

        for action, atoms in enumerate(critic):
            y = 206 + 93 * action
            color = "#0369a1" if action == 0 else "#b45309"
            mean = sum(atoms) / k
            text(left + 22, y + 6, "AB"[action])
            elements.append(f'<line x1="{axis_left}" y1="{y}" '
                            f'x2="{axis_left + axis_width}" y2="{y}" class="axis"/>')
            for atom in atoms:
                x = x_position(atom)
                elements.append(f'<circle cx="{x:.2f}" cy="{y}" r="6" '
                                f'fill="{color}"/>')
            x = x_position(mean)
            elements.append(f'<path d="M{x:.2f},{y-10} l7,10 l-7,10 l-7,-10 Z" '
                            f'fill="{color}" stroke="#fff" stroke-width="1"/>')
            text(left + 67, y + 31, f"atoms: {atoms[0]}, {atoms[1]}", "small")
        means = [sum(atoms) / k for atoms in critic]
        text(left + 20, 396, f"Means: A = {means[0]}; B = {means[1]}", "small")
        text(left + 20, 424, f"Strict greedy gap = {construction['gap']}", "small")

    elements.append('<path d="M474,151 L521,151" class="arrow"/>')
    elements.append('<path d="M747,447 L747,473 L253,473 L253,449" class="arrow"/>')
    text(499, 144, "F", "small", "middle")
    text(499, 494, "F(Q_A) = Q_B; F(Q_B) = Q_A", "body", "middle")
    text(32, 526, "Circles: midpoint quantiles. Diamonds: means used for greedy control.", "small")
    elements.append('<rect x="32" y="546" width="936" height="120" rx="10" '
                    'fill="#ecfdf5" stroke="#a7f3d0"/>')
    text(52, 577, "True expected values (not projected means)", "heading")
    text(52, 608, f"Optimal continuation: Q*(A) = {va}; Q*(B) = {optimal_qb}; "
         f"gap = {va - optimal_qb}")
    text(52, 641, f"Always A: V = {va}. Always B: V = {vb}. "
         f"Stationary-policy loss = {va - vb}.")
    elements.append('</svg>')
    return "\n".join(elements) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check the committed SVG without writing.")
    args = parser.parse_args()
    svg = render()
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != svg:
            raise SystemExit("Figure differs; run python -B tools/render_cycle.py")
        print("Figure matches the exact construction.")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(svg, encoding="utf-8")
        print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
