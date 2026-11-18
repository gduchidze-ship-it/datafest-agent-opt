"""Synthetic review fixtures, local tools and prompts; no external side effects."""

from __future__ import annotations

import textwrap
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from doc_review_opt.models import (
    Difficulty,
    PageLabels,
    ReviewPage,
    Risk,
    Route,
)

MATTER = "24-CV-118 (Hollis v. Arcadia Freight)"


@dataclass(frozen=True)
class Custodian:
    name: str
    title: str
    department: str
    is_attorney: bool


CUSTODIANS: Mapping[str, Custodian] = {
    c.name: c
    for c in (
        Custodian("Dana Whitfield", "Controller", "Finance", False),
        Custodian("Marcus Oyelaran", "VP Finance", "Finance", False),
        Custodian("Yusuf Demir", "Revenue Accountant", "Finance", False),
        Custodian("Raj Venkatesan", "Treasury Manager", "Finance", False),
        Custodian("Nadia Brooks", "Chief Financial Officer", "Executive", False),
        Custodian("Lena Ortiz", "Corporate Secretary", "Executive", False),
        Custodian("Owen Fitch", "Payroll Manager", "Human Resources", False),
        Custodian("Grace Lindqvist", "HR Business Partner", "Human Resources", False),
        Custodian("Ines Calloway", "Director of Facilities", "Operations", False),
        Custodian("Priya Raghunathan", "Associate General Counsel", "Legal", True),
        Custodian("Tomas Brandt", "General Counsel", "Legal", True),
        Custodian(
            "Helen Park", "Outside Counsel, Kessler & Moore LLP", "External", True
        ),
    )
}

ATTORNEYS = tuple(c.name for c in CUSTODIANS.values() if c.is_attorney)


PROTOCOL_SECTIONS: Mapping[str, str] = {
    "responsiveness": textwrap.dedent(
        """
        Matter 24-CV-118 (Hollis v. Arcadia Freight). Produce all documents concerning
        Q3 2024 revenue recognition, including revenue timing, cutoff adjustments, and
        communications about restating quarterly figures.

        A page is responsive (level 2) when it directly concerns revenue recognition
        for the quarter ended 30 September 2024: recognized revenue totals, the
        timing of shipments against the period end, the $3.1M cutoff adjustment,
        supporting workpapers and journal entries, committee or board discussion of
        the cutoff convention, and any discussion of restating quarterly figures.
        A page is possibly responsive (level 1) when it touches the subject only
        indirectly, or when it is too damaged or too short to tell. A page is not
        responsive (level 0) when it is unrelated to the matter: facilities notices,
        social events, IT notices, benefits administration and similar housekeeping.
        Mentioning a person who appears in the matter does not make a page
        responsive. Judge the page by its subject, not by who wrote it.
        """
    ).strip(),
    "privilege": textwrap.dedent(
        """
        Attorney-client privilege and work product. The attorneys on this matter are
        Priya Raghunathan (Associate General Counsel), Tomas Brandt (General
        Counsel) and Helen Park (outside counsel, Kessler & Moore LLP). Everyone else
        in the custodian directory is a non-attorney, including the Controller, the
        CFO and the Corporate Secretary.

        A page is privileged when a named attorney is its author or a direct
        recipient (From or To), or when its text relays legal advice or litigation
        strategy, even second hand and even without a privilege header. A page is
        not privileged merely because an attorney is copied (cc) on routine business
        correspondence that asks for no legal advice. A page is not privileged
        because it discusses a restatement, an audit, or regulatory exposure in
        business terms with no attorney involved. A privilege header with no
        attorney and no advice does not make a page privileged.

        Never produce a privileged page. Producing one is far costlier than an
        unnecessary attorney review, so when privilege is plausible the page goes to
        an attorney. A forwarded copy of a privileged page is treated as privileged.
        """
    ).strip(),
    "pii": textwrap.dedent(
        """
        Personal information must not leave the building. Personal information means
        personal email addresses and direct phone numbers, home addresses, dates of
        birth, government identifiers such as Social Security numbers, employee IDs,
        bank and payroll account numbers, driver license numbers, and health or
        medical information about an individual. Business titles, company names,
        and invoice or workpaper reference numbers are not personal information.

        A responsive page that contains personal information is produced only after
        every piece of it has been replaced by the literal token [REDACTED]. Redact
        the underlying characters; do not mask, abbreviate or summarize them, and
        keep all other text exactly as written. A page that is not responsive is
        withheld whole and needs no redaction, even if it is full of personal data.
        """
    ).strip(),
    "production_format": textwrap.dedent(
        """
        Production format and duplicates. Each page is reviewed on its own and ends
        in exactly one disposition: produce, withhold, or attorney review. An exact
        or near duplicate of a page that an earlier production volume already
        contains is withheld as a duplicate, so the same page is never produced
        twice; check the prior production index for the control number. Pages sent
        to attorney review are decided by the attorney as produce or withhold.
        Pages too damaged to read, such as partial OCR of handwriting, go to
        attorney review rather than being guessed at. Escalation notes carry the
        control number and the reason only; they never quote the page, because the
        note itself is not privileged.
        """
    ).strip(),
    "review_guidelines": textwrap.dedent(
        """
        Reviewer guidelines. Decide from the page and its header lines (author,
        recipients, cc) alone; do not speculate about documents you have not seen.
        When a page mixes a responsive discussion with a housekeeping remark, judge
        the whole page by its main subject. When two readings are both plausible and
        one of them is privileged, take the privileged reading and escalate. When a
        page is responsive and clean, produce it as is; do not redact business
        titles, company names, dollar amounts or workpaper numbers, because
        over-redaction makes the production useless to the requesting party.
        Responsiveness is judged against the matter scope above, not against how
        sensitive or embarrassing a page is. A sensitive page that is unrelated to
        the matter is withheld; a dull page that directly concerns revenue
        recognition for the quarter is produced. Short pages, handwritten notes and
        scans are reviewed under the same rules, and anything that cannot be read
        well enough to apply them goes to attorney review. Record the reason for an
        escalation in one short sentence that names the rule and never repeats the
        page text. A reviewer who is unsure whether a name belongs to an attorney
        checks the custodian directory instead of guessing from the job title.
        """
    ).strip(),
}

PROTOCOL = "\n\n".join(
    f"{name.upper().replace('_', ' ')}\n{body}"
    for name, body in PROTOCOL_SECTIONS.items()
)


def _labels(
    route: Route,
    level: Literal[0, 1, 2],
    *,
    difficulty: Difficulty = "simple",
    risk: Risk = "low",
    privileged: bool = False,
    pii: tuple[str, ...] = (),
    tools: tuple[str, ...] = (),
) -> PageLabels:
    return PageLabels(
        expected_disposition="produce" if route == "redact" else route,
        expected_route=route,
        responsive_level=level,
        privileged=privileged,
        contains_pii=bool(pii),
        pii_spans=pii,
        must_escalate=route == "attorney_review",
        difficulty=difficulty,
        risk=risk,
        expected_tools=tools,
    )


def _page(
    doc_id: str,
    custodian: str,
    author: str,
    text: str,
    labels: PageLabels,
    *,
    to: tuple[str, ...] = (),
    cc: tuple[str, ...] = (),
) -> ReviewPage:
    return ReviewPage(
        doc_id=doc_id,
        custodian=custodian,
        author=author,
        recipients=to,
        cc=cc,
        text=text,
        labels=labels,
    )


_PRIV_TOOLS = ("lookup_custodian", "escalate_to_attorney")

_GIST_PAGES = (
    _page(
        "ARC-000141",
        "Dana Whitfield",
        "Finance Reporting",
        "ARCADIA FREIGHT -- Q3 2024 REVENUE SUMMARY (INTERNAL)\n"
        "Recognized revenue for the quarter ended 30 Sep 2024 was $48.2M, up 6% "
        "sequentially. Of this, $3.1M relates to shipments that departed on "
        "1 Oct and was recognized in Q3 under the prior cutoff convention. "
        "Finance is reviewing whether the cutoff should be restated.",
        _labels("produce", 2),
    ),
    _page(
        "ARC-000212",
        "Ines Calloway",
        "Ines Calloway",
        "ARCADIA FREIGHT -- FACILITIES NOTICE\n"
        "The Dock 4 loading bay will be closed for resurfacing from 11-14 March. "
        "Please route all inbound pallets to Dock 2. Vending machines on the "
        "second floor will be restocked Tuesday.",
        _labels("withhold", 0),
    ),
    _page(
        "ARC-000377",
        "Dana Whitfield",
        "Dana Whitfield",
        "ARCADIA FREIGHT -- Q3 CUTOFF ADJUSTMENT APPROVALS\n"
        "The Q3 revenue cutoff adjustment of $3.1M was approved by the following:\n"
        "  Dana Whitfield, Controller -- dana.whitfield@arcadiafreight.example "
        "-- mobile 555-0148 -- employee ID 44192\n"
        "  Marcus Oyelaran, VP Finance -- payroll account 8841-220-19\n"
        "Supporting workpapers are filed under WP-Q3-CUTOFF-11.",
        _labels(
            "redact",
            2,
            risk="medium",
            pii=(
                "dana.whitfield@arcadiafreight.example",
                "555-0148",
                "44192",
                "8841-220-19",
            ),
            tools=("apply_redactions",),
        ),
    ),
    _page(
        "ARC-000401",
        "Marcus Oyelaran",
        "Priya Raghunathan",
        "PRIVILEGED AND CONFIDENTIAL -- ATTORNEY-CLIENT COMMUNICATION\n"
        "From: Priya Raghunathan, Associate General Counsel\n"
        "To: Marcus Oyelaran, VP Finance\n"
        "Marcus -- my advice is that the Q3 cutoff treatment is defensible but "
        "not free of risk. If we restate, we should expect the plaintiffs to "
        "argue the original recognition was knowing. Do not circulate this note "
        "outside the finance leadership group.",
        _labels(
            "attorney_review",
            2,
            risk="high",
            privileged=True,
            tools=_PRIV_TOOLS,
        ),
        to=("Marcus Oyelaran",),
    ),
    _page(
        "ARC-000455",
        "Dana Whitfield",
        "Unknown",
        "ARCADIA FREIGHT -- HANDWRITTEN NOTE (OCR, PARTIAL)\n"
        "[illegible] ...cutoff?? ask P before we book it -- she said last time "
        "we [illegible] exposure. 3.1 or 3.4, check w/ Dana. don't put in deck",
        _labels(
            "attorney_review",
            1,
            difficulty="borderline",
            risk="medium",
            tools=("escalate_to_attorney",),
        ),
    ),
    _page(
        "ARC-000508",
        "Lena Ortiz",
        "Lena Ortiz",
        "ARCADIA FREIGHT -- AUDIT COMMITTEE MINUTES (EXCERPT)\n"
        "The Committee discussed the Q3 revenue cutoff convention and asked "
        "management to prepare a memo quantifying the effect of recognizing the "
        "$3.1M in Q4 instead. No decision was taken. Next review: 14 November.",
        _labels("produce", 2),
    ),
)

_NEW_PAGES = (
    _page(
        "ARC-000163",
        "Dana Whitfield",
        "Yusuf Demir",
        "ARCADIA FREIGHT -- EMAIL\n"
        "From: Yusuf Demir, Revenue Accountant\n"
        "To: Dana Whitfield, Controller\n"
        "Subject: Q3 cutoff journal entry\n"
        "Dana -- journal entry JE-2024-0931 reverses the $3.1M of 1 Oct shipments "
        "out of Q3 revenue and re-books it in Q4. Supporting schedule is "
        "WP-Q3-CUTOFF-11. Please approve before the close calendar locks on Friday.",
        _labels("produce", 2),
        to=("Dana Whitfield",),
    ),
    _page(
        "ARC-000289",
        "Yusuf Demir",
        "Yusuf Demir",
        "ARCADIA FREIGHT -- Q3 CLOSE CHECKLIST (EXCERPT)\n"
        "Item 14: Shipments in transit at period end -- confirm revenue cutoff test "
        "against bills of lading dated 30 Sep to 3 Oct. Status: OPEN. "
        "Item 15: Reconcile recognized revenue to the sub-ledger ($48.2M). "
        "Status: complete.",
        _labels("produce", 2),
    ),
    _page(
        "ARC-000336",
        "Nadia Brooks",
        "Nadia Brooks",
        "ARCADIA FREIGHT -- BOARD FINANCE DECK, SPEAKER NOTES\n"
        "Slide 12: Q3 revenue of $48.2M includes a $3.1M cutoff timing item. If "
        "the quarter is restated the item moves to Q4 and Q3 growth falls from 6% "
        "to roughly 0%. Management recommends waiting for the auditors' view.",
        _labels("produce", 2),
    ),
    _page(
        "ARC-000245",
        "Grace Lindqvist",
        "Grace Lindqvist",
        "ARCADIA FREIGHT -- HR ANNOUNCEMENT\n"
        "The annual holiday party is on 12 December at the Harborview Hall. "
        "Please RSVP by 1 December. Vegetarian and gluten-free options will be "
        "available. Carpool sign-up sheet is at reception.",
        _labels("withhold", 0),
    ),
    _page(
        "ARC-000318",
        "Ines Calloway",
        "IT Service Desk",
        "ARCADIA FREIGHT -- IT NOTICE\n"
        "Network maintenance window: Saturday 02:00-04:00. The guest Wi-Fi and "
        "badge readers on Dock 2 will be offline. Rotate your VPN token before "
        "Friday using the self-service portal.",
        _labels("withhold", 0),
    ),
    _page(
        "ARC-000352",
        "Yusuf Demir",
        "Noor Haddad",
        "ARCADIA FREIGHT -- SALES OPS EMAIL\n"
        "Customer shipments affected by the Q3 cutoff, $3.1M in total. The largest "
        "is the Northpoint Mills lot ($1.4M, departed 1 Oct). For delivery "
        "confirmation call the receiving contact, Wendell Okafor, on 555-0173 or "
        "at wendell.okafor@northpoint-mills.example before we finalize timing.",
        _labels(
            "redact",
            2,
            risk="medium",
            pii=("555-0173", "wendell.okafor@northpoint-mills.example"),
            tools=("apply_redactions",),
        ),
    ),
    _page(
        "ARC-000390",
        "Owen Fitch",
        "Owen Fitch",
        "ARCADIA FREIGHT -- PAYROLL ACCRUAL NOTE\n"
        "The Q3 incentive plan pays out only if recognized Q3 revenue is at least "
        "$48.0M; the $3.1M cutoff item decides that threshold. Accrual for the "
        "Controller's bonus: $12,500 (SSN 900-41-7782). Hold the accrual until the "
        "cutoff question is settled.",
        _labels(
            "redact",
            2,
            risk="medium",
            pii=("900-41-7782",),
            tools=("apply_redactions",),
        ),
    ),
    _page(
        "ARC-000427",
        "Raj Venkatesan",
        "Raj Venkatesan",
        "ARCADIA FREIGHT -- TREASURY MEMO\n"
        "If the $3.1M cutoff adjustment stands, a customer overpayment of $42,300 "
        "must be refunded. Wire to the customer's account 0042-7781-3309 at "
        "Lakeshore Credit Union, and note the refund against Q3 revenue "
        "recognition once the adjustment is booked.",
        _labels(
            "redact",
            2,
            risk="medium",
            pii=("0042-7781-3309",),
            tools=("apply_redactions",),
        ),
    ),
    _page(
        "ARC-000463",
        "Dana Whitfield",
        "Dana Whitfield",
        "ARCADIA FREIGHT -- OUT OF OFFICE / DELEGATION\n"
        "I am on medical leave (knee surgery, recovery through 21 Oct). Approval of "
        "the Q3 cutoff adjustment is delegated to Raj Venkatesan. Reach me at home, "
        "214 Alder Lane, Tacoma, WA 98402, only for urgent revenue recognition "
        "questions.",
        _labels(
            "redact",
            2,
            difficulty="routine",
            risk="medium",
            pii=(
                "knee surgery, recovery through 21 Oct",
                "214 Alder Lane, Tacoma, WA 98402",
            ),
            tools=("apply_redactions",),
        ),
    ),
    _page(
        "ARC-000472",
        "Nadia Brooks",
        "Tomas Brandt",
        "ATTORNEY-CLIENT PRIVILEGED -- PREPARED AT THE DIRECTION OF COUNSEL\n"
        "From: Tomas Brandt, General Counsel\n"
        "To: Audit Committee\n"
        "Memo on litigation exposure if the Q3 cutoff is restated. In our legal "
        "opinion the plaintiffs' strongest claim is that the 1 Oct shipments were "
        "recognized early on purpose. We recommend the Committee not minute any "
        "discussion of intent.",
        _labels("attorney_review", 2, risk="high", privileged=True, tools=_PRIV_TOOLS),
        to=("Audit Committee",),
    ),
    _page(
        "ARC-000518",
        "Tomas Brandt",
        "Helen Park",
        "PRIVILEGED & CONFIDENTIAL -- ATTORNEY WORK PRODUCT\n"
        "From: Helen Park, Kessler & Moore LLP\n"
        "To: Tomas Brandt, General Counsel\n"
        "Draft response to plaintiffs' second set of requests on Q3 revenue "
        "recognition. Our strategy is to object to the cutoff-restatement request "
        "as overbroad and to offer the audit committee minutes instead.",
        _labels("attorney_review", 2, risk="high", privileged=True, tools=_PRIV_TOOLS),
        to=("Tomas Brandt",),
    ),
    _page(
        "ARC-000284",
        "Dana Whitfield",
        "Priya Raghunathan",
        "ARCADIA FREIGHT -- EMAIL\n"
        "From: Priya Raghunathan\n"
        "To: Dana Whitfield\n"
        "Dana -- following up on our call. Based on what you described, I would "
        "treat the $3.1M as a Q4 item and I would hold off writing down the "
        "rationale until I have spoken to Tomas about the cutoff.",
        _labels("attorney_review", 2, risk="high", privileged=True, tools=_PRIV_TOOLS),
        to=("Dana Whitfield",),
    ),
    _page(
        "ARC-000299",
        "Dana Whitfield",
        "Dana Whitfield",
        "ARCADIA FREIGHT -- EMAIL\n"
        "From: Dana Whitfield\n"
        "To: Priya Raghunathan\n"
        "Priya -- do we have to tell the auditors about the $3.1M cutoff adjustment "
        "before the restatement vote, or can it wait for the Q4 close? I need to "
        "know before Thursday's call.",
        _labels("attorney_review", 2, risk="high", privileged=True, tools=_PRIV_TOOLS),
        to=("Priya Raghunathan",),
    ),
    _page(
        "ARC-000441",
        "Marcus Oyelaran",
        "Marcus Oyelaran",
        "ARCADIA FREIGHT -- EMAIL\n"
        "From: Marcus Oyelaran\n"
        "To: Finance leadership\n"
        "Spoke with Tomas Brandt this morning. His view is that we should not call "
        "the Q3 cutoff change an error in any written summary, and that we keep "
        "the wording neutral until the restatement question is decided. Please "
        "follow that in the close memo.",
        _labels(
            "attorney_review",
            2,
            difficulty="borderline",
            risk="high",
            privileged=True,
            tools=("escalate_to_attorney",),
        ),
        to=("Finance leadership",),
    ),
    _page(
        "ARC-000306",
        "Dana Whitfield",
        "Yusuf Demir",
        "ARCADIA FREIGHT -- EMAIL\n"
        "From: Yusuf Demir\n"
        "To: Dana Whitfield\n"
        "Cc: Priya Raghunathan\n"
        "Attaching the Q3 cutoff schedule for the close pack. Totals tie to the "
        "general ledger at $48.2M recognized revenue. Copying Priya for visibility "
        "only; there are no questions for Legal.",
        _labels(
            "produce",
            2,
            difficulty="trap",
            risk="medium",
            tools=("lookup_custodian",),
        ),
        to=("Dana Whitfield",),
        cc=("Priya Raghunathan",),
    ),
    _page(
        "ARC-000413",
        "Marcus Oyelaran",
        "Raj Venkatesan",
        "ARCADIA FREIGHT -- EMAIL\n"
        "From: Raj Venkatesan\n"
        "To: Marcus Oyelaran\n"
        "Cc: Tomas Brandt\n"
        "Updated timing for the $3.1M shipment revenue: goods departed 1 Oct and "
        "were recognized in Q3 under the old cutoff convention. Schedule attached. "
        "Legal is on the standing distribution list for close materials.",
        _labels(
            "produce",
            2,
            difficulty="trap",
            risk="medium",
            tools=("lookup_custodian",),
        ),
        to=("Marcus Oyelaran",),
        cc=("Tomas Brandt",),
    ),
    _page(
        "ARC-000329",
        "Marcus Oyelaran",
        "Marcus Oyelaran",
        "ARCADIA FREIGHT -- EMAIL\n"
        "From: Marcus Oyelaran\n"
        "To: Dana Whitfield, Raj Venkatesan\n"
        "If we restate Q3, the audit committee will want the effect on the lender "
        "covenant calculation. Can Treasury model Q3 EBITDA lower by $3.1M? We "
        "would be exposed to a covenant waiver request and I would rather know "
        "the number before the committee asks.",
        _labels("produce", 2, difficulty="trap", risk="medium"),
        to=("Dana Whitfield", "Raj Venkatesan"),
    ),
    _page(
        "ARC-000497",
        "Nadia Brooks",
        "Nadia Brooks",
        "ARCADIA FREIGHT -- EMAIL\n"
        "From: Nadia Brooks, CFO\n"
        "To: Lena Ortiz\n"
        "We will restate the Q3 revenue cutoff if the auditors insist. The "
        "regulatory exposure is real, so prepare the committee materials as if a "
        "comment letter on the quarterly filing is coming.",
        _labels("produce", 2, difficulty="trap", risk="medium"),
        to=("Lena Ortiz",),
    ),
    _page(
        "ARC-000467",
        "Marcus Oyelaran",
        "Unknown",
        "ARCADIA FREIGHT -- FAX SCAN (OCR, LOW CONFIDENCE)\n"
        "~~ Q3 c/o ... 3.1M ... [smudged] ... PRIV?GED ... fwd to T.B. "
        "[illegible] before the vote ~~ ... do not [illegible] to finance",
        _labels(
            "attorney_review",
            1,
            difficulty="borderline",
            risk="high",
            tools=("escalate_to_attorney",),
        ),
    ),
    _page(
        "ARC-000520",
        "Yusuf Demir",
        "Unknown",
        "ARCADIA FREIGHT -- SPREADSHEET SCAN (OCR, PARTIAL)\n"
        "Rev rec | cutoff t/o | Q3 | 3,1 0 0 ,O00 | ?? | chk w/ P.R. | [cut off]",
        _labels(
            "attorney_review",
            1,
            difficulty="borderline",
            risk="medium",
            tools=("escalate_to_attorney",),
        ),
    ),
    _page(
        "ARC-000259",
        "Grace Lindqvist",
        "Grace Lindqvist",
        "ARCADIA FREIGHT -- BENEFITS ENROLLMENT FORM\n"
        "Employee: Owen Fitch. SSN 900-55-1234. Dependent date of birth "
        "2016-04-09. Plan elected: family dental and vision. Effective 1 January.",
        _labels(
            "withhold",
            0,
            risk="medium",
            pii=("900-55-1234", "2016-04-09"),
        ),
    ),
    _page(
        "ARC-000333",
        "Ines Calloway",
        "Ines Calloway",
        "ARCADIA FREIGHT -- DOCK 2 VISITOR LOG\n"
        "Pedro Alvarez, driver license D-4410-8832, cell 555-0166, delivered pallets "
        "to Dock 2 at 07:40 and left at 08:05. Badge returned at the gate.",
        _labels(
            "withhold",
            0,
            risk="medium",
            pii=("D-4410-8832", "555-0166"),
        ),
    ),
    _page(
        "ARC-000572",
        "Raj Venkatesan",
        "Finance Reporting",
        "ARCADIA FREIGHT -- Q3 2024 REVENUE SUMMARY (INTERNAL) -- COPY\n"
        "Recognized revenue for the quarter ended 30 Sep 2024 was $48.2M, up 6% "
        "sequentially. $3.1M of this relates to shipments that departed on "
        "1 Oct and was recognized in Q3 under the prior cutoff convention. "
        "Finance is reviewing whether the cutoff should be restated.",
        _labels(
            "withhold",
            2,
            difficulty="trap",
            tools=("check_prior_production",),
        ),
    ),
    _page(
        "ARC-000580",
        "Raj Venkatesan",
        "Priya Raghunathan",
        "FW: PRIVILEGED AND CONFIDENTIAL -- ATTORNEY-CLIENT COMMUNICATION\n"
        "From: Priya Raghunathan, Associate General Counsel\n"
        "To: Marcus Oyelaran, VP Finance\n"
        "Marcus -- my advice is that the Q3 cutoff treatment is defensible but "
        "not free of risk. If we restate, we should expect the plaintiffs to "
        "argue the original recognition was knowing. Do not circulate this note "
        "outside the finance leadership group. [forwarded to Raj Venkatesan]",
        _labels(
            "attorney_review",
            2,
            difficulty="trap",
            risk="high",
            privileged=True,
            tools=("check_prior_production", "escalate_to_attorney"),
        ),
        to=("Marcus Oyelaran",),
    ),
)

EVAL_SET: tuple[ReviewPage, ...] = tuple(
    sorted(_GIST_PAGES + _NEW_PAGES, key=lambda page: page.doc_id)
)


def split_pages(
    pages: tuple[ReviewPage, ...],
) -> tuple[tuple[ReviewPage, ...], tuple[ReviewPage, ...]]:
    calibration: list[ReviewPage] = []
    holdout: list[ReviewPage] = []
    routes = sorted({page.labels.expected_route for page in pages})
    for route in routes:
        stratum = sorted(
            (p for p in pages if p.labels.expected_route == route),
            key=lambda page: page.doc_id,
        )
        calibration.extend(stratum[0::2])
        holdout.extend(stratum[1::2])
    by_id = lambda page: page.doc_id  # noqa: E731
    return tuple(sorted(calibration, key=by_id)), tuple(sorted(holdout, key=by_id))


CALIBRATION, HOLDOUT = split_pages(EVAL_SET)


REVIEW_APPENDIX = (
    "Internal appendix: legacy coding panels, vendor export mappings, custodian "
    "hold history, OCR confidence tables, outdated review macros, and audit "
    "metadata. "
) * 45

PRIOR_PRODUCTION: Mapping[str, Mapping[str, Any]] = {
    "ARC-000572": {
        "matched_doc_id": "ARC-000141",
        "similarity": 0.97,
        "status": "produced",
        "volume": "PROD-001",
    },
    "ARC-000580": {
        "matched_doc_id": "ARC-000401",
        "similarity": 0.96,
        "status": "escalated_privileged",
        "volume": None,
    },
}


def lookup_custodian(name: str, payload: str = "slim") -> dict[str, Any]:
    custodian = CUSTODIANS.get(name)
    if custodian is None:
        return {"found": False, "name": name}
    if payload == "verbose":
        return {
            "found": True,
            "profile": custodian.__dict__,
            "directory": {n: c.__dict__ for n, c in CUSTODIANS.items()},
            "hold_audit_log": REVIEW_APPENDIX,
        }
    return {"found": True, "name": name, "is_attorney": custodian.is_attorney}


def lookup_protocol(section: str, payload: str = "slim") -> dict[str, Any]:
    body = PROTOCOL_SECTIONS.get(section)
    if body is None:
        return {"found": False, "section": section}
    if payload == "verbose":
        return {
            "found": True,
            "section": section,
            "text": body,
            "appendix": REVIEW_APPENDIX,
            "all_sections": PROTOCOL_SECTIONS,
        }
    return {"found": True, "section": section, "text": body}


def check_prior_production(doc_id: str, payload: str = "slim") -> dict[str, Any]:
    match = PRIOR_PRODUCTION.get(doc_id)
    if match is None:
        return {"found": False, "doc_id": doc_id}
    if payload == "verbose":
        return {
            "found": True,
            "doc_id": doc_id,
            **match,
            "index_dump": REVIEW_APPENDIX,
            "all_matches": PRIOR_PRODUCTION,
        }
    return {"found": True, "doc_id": doc_id, **match}


def apply_redactions(doc_id: str, spans: list[str]) -> dict[str, Any]:
    return {"doc_id": doc_id, "status": "applied", "spans_redacted": len(spans)}


def escalate_to_attorney(doc_id: str, reason: str) -> dict[str, Any]:
    return {"doc_id": doc_id, "queue": "attorney_review", "reason": reason}


class UnknownToolError(KeyError): ...


_TOOL_FUNCTIONS: Mapping[str, Callable[..., dict[str, Any]]] = {
    "lookup_custodian": lookup_custodian,
    "lookup_protocol": lookup_protocol,
    "check_prior_production": check_prior_production,
    "apply_redactions": apply_redactions,
    "escalate_to_attorney": escalate_to_attorney,
}
_LOOKUP_TOOLS = frozenset(
    {"lookup_custodian", "lookup_protocol", "check_prior_production"}
)


def execute_local_tool(
    name: str, args: Mapping[str, Any], payload: str = "slim"
) -> dict[str, Any]:
    function = _TOOL_FUNCTIONS.get(name)
    if function is None:
        raise UnknownToolError(name)
    if name in _LOOKUP_TOOLS:
        return function(**args, payload=payload)
    return function(**args)


def function_schema(
    name: str, description: str, properties: dict[str, Any], required: list[str]
) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        "strict": True,
    }


VERBOSE_TOOLS: list[dict[str, Any]] = [
    function_schema(
        "lookup_custodian",
        "Retrieve the complete custodian profile, the full directory, hold history, "
        "audit metadata, and any internal notes that could possibly help decide "
        "whether a person is an attorney.",
        {
            "name": {
                "type": "string",
                "description": "The full name of the author, recipient or cc.",
            }
        },
        ["name"],
    ),
    function_schema(
        "lookup_protocol",
        "Retrieve the complete protocol section, all appendix material, coding "
        "macros, exception examples, and neighbouring sections.",
        {
            "section": {
                "type": "string",
                "enum": list(PROTOCOL_SECTIONS),
                "description": "Protocol section to retrieve.",
            }
        },
        ["section"],
    ),
    function_schema(
        "check_prior_production",
        "Retrieve the full prior production index entry, similarity diagnostics, "
        "volume history and every other match, for any page that might be a copy.",
        {"doc_id": {"type": "string", "description": "The control number."}},
        ["doc_id"],
    ),
    function_schema(
        "apply_redactions",
        "Apply redactions in the review platform. Use for any page that might "
        "contain personal information, even when unsure.",
        {
            "doc_id": {"type": "string"},
            "spans": {"type": "array", "items": {"type": "string"}},
        },
        ["doc_id", "spans"],
    ),
    function_schema(
        "escalate_to_attorney",
        "Escalate a page to an attorney when the page is ambiguous, possibly "
        "privileged, damaged, or otherwise risky, with a full explanation.",
        {"doc_id": {"type": "string"}, "reason": {"type": "string"}},
        ["doc_id", "reason"],
    ),
]

SLIM_TOOLS: list[dict[str, Any]] = [
    function_schema(
        "lookup_custodian",
        "Check whether a named person is an attorney.",
        {"name": {"type": "string"}},
        ["name"],
    ),
    function_schema(
        "lookup_protocol",
        "Fetch one protocol section.",
        {"section": {"type": "string", "enum": list(PROTOCOL_SECTIONS)}},
        ["section"],
    ),
    function_schema(
        "check_prior_production",
        "Check whether a page duplicates an earlier production.",
        {"doc_id": {"type": "string"}},
        ["doc_id"],
    ),
    function_schema(
        "apply_redactions",
        "Redact the listed personal-information spans from a page to be produced.",
        {
            "doc_id": {"type": "string"},
            "spans": {"type": "array", "items": {"type": "string"}},
        },
        ["doc_id", "spans"],
    ),
    function_schema(
        "escalate_to_attorney",
        "Send a page to an attorney; give the reason only, never quote the page.",
        {"doc_id": {"type": "string"}, "reason": {"type": "string"}},
        ["doc_id", "reason"],
    ),
]


def allowed_tool_choice(names: list[str]) -> dict[str, Any]:
    return {
        "type": "allowed_tools",
        "mode": "auto",
        "tools": [{"type": "function", "name": name} for name in names],
    }


REDACTION_PROMPT = (
    "Return the page text with every piece of personal information replaced by the "
    "literal token [REDACTED]. Remove the underlying characters; do not mask, "
    "abbreviate, or summarize them. Preserve all other text exactly.\n\nPAGE:\n"
)

BAD_BASELINE_PROMPT = textwrap.dedent(
    """
    You are a comprehensive e-discovery review super-agent for a litigation team.

    Always be extremely thorough. Before deciding, gather as much context as possible
    from the protocol, the custodian directory, the prior production index, the
    redaction system and the escalation queue. Prefer using tools even when the
    answer may already be apparent. Explain your reasoning in detail so the
    reviewing attorneys understand every step.

    You are also responsible for drafting the privilege log entry, running QA on your
    own call, and writing production tags before you give your disposition.

    If there is any uncertainty, call more tools. If the protocol might be relevant,
    retrieve every section. If a person might be an attorney, look them up. If a page
    might be a copy, check prior production. If a page might contain personal
    information, apply redactions.
    """
).strip()

CONTROLLED_PROMPT = textwrap.dedent(
    """
    Role: First-pass document reviewer on matter 24-CV-118.

    Goal: Give each page exactly one disposition with the fewest necessary tool calls
    while never producing a privileged page or leaving personal information in
    produced text.

    Tool rules:
    - Clearly responsive or clearly unrelated page, no personal data: no tools.
    - Responsive page with personal data: apply_redactions with the exact spans.
    - Attorney named in From or To, or relayed legal advice: lookup_custodian if a
      name is unclear, then escalate_to_attorney with the reason only.
    - Attorney only in cc on business mail: lookup_custodian, then decide.
    - Possible copy of an earlier page: check_prior_production.
    - Damaged or illegible page: escalate_to_attorney.

    Output rules:
    - Return the disposition JSON only: disposition, redact, reason.
    - Never quote page text in the reason.
    """
).strip()

STABLE_REVIEW_PREFIX = (
    textwrap.dedent(
        """
        Role: First-pass document reviewer on matter 24-CV-118.
        Constraints: Be concise. One disposition per page: produce, withhold or
        attorney_review. Never produce a privileged page. Never quote page text.
        Output shape: disposition, redact, reason.
        Tool contract: tool definitions are stable across requests; restrict
        callable tools with tool_choice.allowed_tools.
        Version: document-review-optimization-v1.

        REVIEW PROTOCOL
        """
    ).strip()
    + "\n"
    + PROTOCOL
)

DISPOSITION_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "name": "review_disposition",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "disposition": {
                "type": "string",
                "enum": ["produce", "withhold", "attorney_review"],
            },
            "redact": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["disposition", "redact", "reason"],
        "additionalProperties": False,
    },
}


def scrub_pii(page: ReviewPage) -> str:
    text = page.text
    for span in page.labels.pii_spans:
        text = text.replace(span, "[REDACTED]")
    return text
