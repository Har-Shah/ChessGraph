"""Grounding evaluation: does every claim in the report hold up?

Three checks, in increasing strength.

1. RESOLUTION. Every citation points at a move that exists, and the move it
   names matches the move actually played in that game at that ply.

2. SUPPORT. Every citation is actually an instance of what the claim asserts.
   A claim about forks must cite moves labelled with the fork theme. A claim
   about the Caro-Kann must cite games in the Caro-Kann. This catches wiring
   bugs where a report cites real but irrelevant moves, which is the failure
   mode that looks most convincing to a reader.

3. NUMERIC FIDELITY. Every number in a claim is recomputed from the database
   and compared. Catches drift between what a claim says and what the data
   supports.

HONEST NOTE ON WHAT THIS PROVES TODAY
The MVP report is template generated, so checks 1 and 3 are close to
tautological: the same query that produced a number is used to verify it. They
are still worth running as regression tests, and they exist now so that the
moment a language model writes the prose there is an established baseline of
100 percent to measure the drop against. Check 2 is not tautological even
today, because citation selection and claim text are produced by separate
queries and can disagree.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from chessgraph.report.generate import PrepReport, Claim
from chessgraph.store.db import Store


@dataclass
class GroundingIssue:
    claim_text: str
    kind: str
    problem: str
    detail: str = ""


@dataclass
class GroundingResult:
    total_claims: int
    total_citations: int
    claims_with_citations: int
    resolved_citations: int
    supported_citations: int
    numeric_checks: int
    numeric_passed: int
    issues: list[GroundingIssue] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "total_claims": self.total_claims,
            "total_citations": self.total_citations,
            "citation_coverage": round(
                self.claims_with_citations / max(self.total_claims, 1), 4),
            "citation_resolution_rate": round(
                self.resolved_citations / max(self.total_citations, 1), 4),
            "citation_support_rate": round(
                self.supported_citations / max(self.total_citations, 1), 4),
            "numeric_fidelity": round(
                self.numeric_passed / max(self.numeric_checks, 1), 4),
            "n_issues": len(self.issues),
        }


def _check_support(store: Store, claim: Claim, cit) -> tuple[bool, str]:
    """Is this citation an instance of what the claim asserts?"""
    ev = claim.evidence
    if claim.kind == "weakness":
        theme = ev.get("theme")
        row = store.one(
            "SELECT 1 FROM move_themes WHERE game_id=? AND ply=? AND theme=?",
            (cit.game_id, cit.ply, theme))
        if row is None:
            return False, f"cited move is not labelled '{theme}'"
        return True, ""
    if claim.kind in ("repertoire", "opening_weakness"):
        opening = ev.get("opening")
        row = store.one("SELECT opening FROM games WHERE game_id=?", (cit.game_id,))
        if row is None:
            return False, "cited game not found"
        if row["opening"] != opening:
            return False, f"cited game is {row['opening']}, claim is about {opening}"
        return True, ""
    return True, ""


def _check_numbers(store: Store, claim: Claim, subject: str) -> list[tuple[bool, str]]:
    """Recompute the claim's numbers from the database."""
    ev, out = claim.evidence, []
    if claim.kind == "weakness":
        row = store.one(
            """SELECT COUNT(*) n, AVG(m.cp_loss) avg_cp
               FROM move_themes t
               JOIN moves m ON m.game_id=t.game_id AND m.ply=t.ply
               JOIN games g ON g.game_id=m.game_id
               WHERE g.subject=? AND m.is_subject_move=1 AND t.theme=?""",
            (subject, ev.get("theme")))
        out.append((row["n"] == ev.get("count"),
                    f"count claimed {ev.get('count')}, actual {row['n']}"))
        claimed, actual = ev.get("avg_cp_loss"), round(row["avg_cp"] or 0)
        out.append((abs(claimed - actual) <= 1,
                    f"avg cp claimed {claimed}, actual {actual}"))
    elif claim.kind == "repertoire":
        row = store.one(
            """SELECT COUNT(*) n, AVG(subject_score) score FROM games
               WHERE subject=? AND subject_color=? AND opening=?""",
            (subject, ev.get("color"), ev.get("opening")))
        out.append((row["n"] == ev.get("games"),
                    f"games claimed {ev.get('games')}, actual {row['n']}"))
    elif claim.kind == "opening_weakness":
        row = store.one(
            """SELECT COUNT(DISTINCT g.game_id) games, AVG(m.cp_loss) acpl
               FROM moves m JOIN games g ON g.game_id=m.game_id
               WHERE g.subject=? AND m.is_subject_move=1 AND g.opening=?
                 AND m.cp_loss IS NOT NULL AND m.judgment!='book'""",
            (subject, ev.get("opening")))
        out.append((row["games"] == ev.get("games"),
                    f"games claimed {ev.get('games')}, actual {row['games']}"))
    return out


def evaluate_grounding(store: Store, report: PrepReport) -> GroundingResult:
    res = GroundingResult(0, 0, 0, 0, 0, 0, 0)
    for claim in report.all_claims():
        res.total_claims += 1
        if claim.citations:
            res.claims_with_citations += 1
        else:
            res.issues.append(GroundingIssue(
                claim.text, claim.kind, "no citations"))

        for cit in claim.citations:
            res.total_citations += 1
            if cit.ply == 0:
                # Game level citation: check the game exists.
                row = store.one("SELECT url FROM games WHERE game_id=?",
                                (cit.game_id,))
                if row and row["url"] == cit.url:
                    res.resolved_citations += 1
                else:
                    res.issues.append(GroundingIssue(
                        claim.text, claim.kind, "unresolved game citation",
                        cit.game_id))
                    continue
            else:
                row = store.one(
                    "SELECT san, cp_loss FROM moves WHERE game_id=? AND ply=?",
                    (cit.game_id, cit.ply))
                if row is None:
                    res.issues.append(GroundingIssue(
                        claim.text, claim.kind, "citation does not resolve",
                        f"{cit.game_id}:{cit.ply}"))
                    continue
                if row["san"] != cit.san:
                    res.issues.append(GroundingIssue(
                        claim.text, claim.kind, "cited move does not match",
                        f"cited {cit.san}, actual {row['san']}"))
                    continue
                res.resolved_citations += 1

            ok, why = _check_support(store, claim, cit)
            if ok:
                res.supported_citations += 1
            else:
                res.issues.append(GroundingIssue(
                    claim.text, claim.kind, "citation does not support claim", why))

        for ok, why in _check_numbers(store, claim, report.subject):
            res.numeric_checks += 1
            if ok:
                res.numeric_passed += 1
            else:
                res.issues.append(GroundingIssue(
                    claim.text, claim.kind, "numeric mismatch", why))
    return res
