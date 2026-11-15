"""Generate ``samples/sample.pdf`` so the CLI and UI can be tried immediately.

    python scripts/make_sample_pdf.py
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

PAGES: list[tuple[str, str]] = [
    (
        "The Economics of Remote Work",
        "Executive summary\n\n"
        "Remote work moved from a perk to a default for many knowledge workers between 2020 "
        "and 2023, and organizations are still negotiating what comes next. This report "
        "examines the measurable effects of remote and hybrid arrangements on productivity, "
        "cost structure, hiring and employee retention.\n\n"
        "Three findings stand out. First, output in task-based roles is broadly unchanged or "
        "slightly higher under hybrid arrangements, while fully remote onboarding is measurably "
        "slower for junior staff. Second, real-estate savings are real but smaller than headline "
        "figures suggest once flexible-office fit-out and travel budgets are included. Third, "
        "the largest gains come from access to a wider hiring pool rather than from direct cost "
        "reduction.\n\n"
        "The recommendation is a structured hybrid model with explicit coordination days, "
        "role-specific expectations, and outcome-based performance metrics that do not depend "
        "on physical presence. Organizations that treat remote work as an all-or-nothing "
        "ideological question consistently underperform those that treat it as an operating "
        "model to be tuned.",
    ),
    (
        "Findings",
        "Productivity\n\n"
        "Studies of task-based work find small positive effects from remote work, driven by "
        "fewer interruptions and reduced commuting time. Effects are weaker or negative for "
        "work that depends on rapid feedback, such as early-career mentoring, complex "
        "negotiations and creative workshops. The variance between teams is larger than the "
        "average effect, which suggests management practice matters more than location.\n\n"
        "Cost structure\n\n"
        "Firms that reduced office space reported savings of 15 to 30 percent of occupancy "
        "cost, but many offset part of that with co-working allowances, home-office stipends "
        "and additional travel for team gatherings. Net savings cluster nearer 10 percent for "
        "organizations that downsized deliberately rather than opportunistically.\n\n"
        "Hiring and retention\n\n"
        "Remote-friendly roles attract a materially larger applicant pool, and time-to-fill "
        "falls in competitive technical markets. Retention improves when remote work is "
        "predictable and documented; it deteriorates when policy changes are announced "
        "abruptly. Compensation bands, not location policy, remain the dominant driver of "
        "voluntary attrition.\n\n"
        "Risks\n\n"
        "The main risks are uneven access to informal information, weaker cross-team "
        "relationships, and compliance exposure when staff work across tax jurisdictions. "
        "Each of these is manageable with explicit process, but none resolves itself.",
    ),
    (
        "Recommendations",
        "1. Adopt a documented hybrid default. Two or three coordination days per week give "
        "teams predictable overlap without pretending that presence is performance.\n\n"
        "2. Measure outcomes, not attendance. Define role-level deliverables and review them "
        "quarterly. Attendance-based proxies for productivity erode trust and hide real "
        "bottlenecks.\n\n"
        "3. Invest in onboarding. Junior employees need structured pairing and visible "
        "progression; the cost of getting this wrong shows up eighteen months later as "
        "attrition.\n\n"
        "4. Right-size the portfolio. Consolidate offices around hubs that people actually "
        "use, and fund the collaboration space that remains properly.\n\n"
        "5. Handle jurisdiction and compliance early. A short policy review is far cheaper "
        "than retroactive tax exposure.\n\n"
        "Conclusion\n\n"
        "Remote work is not a binary choice between full presence and full distance. The "
        "organizations that capture the benefits and avoid the costs are those that treat "
        "location as a design parameter, review the results honestly, and adjust.",
    ),
]


def main() -> None:
    target = Path(__file__).resolve().parents[1] / "samples" / "sample.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)

    document = pymupdf.open()
    for title, body in PAGES:
        page = document.new_page()
        page.insert_text((60, 72), title, fontsize=16, fontname="hebo")
        page.insert_textbox(pymupdf.Rect(60, 92, 535, 780), body, fontsize=10)
    document.set_metadata(
        {"title": "The Economics of Remote Work", "author": "Sample Report", "subject": "Economics"}
    )
    document.save(target)
    document.close()
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
