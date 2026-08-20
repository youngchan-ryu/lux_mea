"""Spoken text -> part id.

Korean is decomposed into jamo before fuzzy matching, so ASR errors that keep
the consonants usually still land on the right part. Scoring is by total length
of the aliases that matched, not by similarity alone -- otherwise a short
generic alias beats a specific one on any long sentence."""
from __future__ import annotations
import re
import time
import unicodedata
from dataclasses import dataclass, field
from rapidfuzz import fuzz

try:
    import hgtk
    def to_jamo(s: str) -> str:
        out = []
        for ch in s:
            try:
                out.append(hgtk.letter.decompose(ch)) if hgtk.checker.is_hangul(ch) else out.append((ch,))
            except Exception:
                out.append((ch,))
        return "".join("".join(t) for t in out)
except Exception:
    _CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
    _JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
    _JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
    def to_jamo(s: str) -> str:
        out = []
        for ch in s:
            code = ord(ch)
            if 0xAC00 <= code <= 0xD7A3:
                i = code - 0xAC00
                out.append(_CHO[i // 588] + _JUNG[(i % 588) // 28] + _JONG[i % 28].strip())
            else:
                out.append(ch)
        return "".join(out)


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"[^0-9a-z가-힣]", "", s)
    return s


_NUM_SLOT = re.compile(r"(\d+)\s*번?\s*(핀|슬롯|포트|레인|채널)")


@dataclass
class Part:
    pid: str
    aliases: list[str]
    _norm: list[str] = field(default_factory=list)
    _jamo: list[str] = field(default_factory=list)

    def build(self):
        cands = [self.pid] + self.aliases
        self._norm = [normalize(c) for c in cands]
        self._jamo = [to_jamo(n) for n in self._norm]
        return self

    def eval(self, q_norm: str, q_jamo: str, hit: float = 85.0):
        """Returns (best_sim, evidence, pos).

        evidence is the summed length of the aliases that matched, so a part that
        matches something long and specific beats one that matched 'power'. pos is
        where the alias first appeared, which breaks ties toward whatever was said
        first.
        """
        best_sim, evidence, pos = 0.0, 0.0, 10 ** 6
        for n, j in zip(self._norm, self._jamo):
            if not n:
                continue
            sc = max(fuzz.partial_ratio(q_norm, n), fuzz.partial_ratio(q_jamo, j))
            if sc > best_sim:
                best_sim = sc
            if sc >= hit:
                evidence += len(n)
                idx = q_norm.find(n)
                if 0 <= idx < pos:
                    pos = idx
        return best_sim, evidence, pos


def load_parts(parts_dict: dict) -> list[Part]:
    out = []
    for pid, spec in parts_dict.items():
        out.append(Part(pid, list(spec.get("aliases", []))).build())
    return out


def load_links(parts_dict: dict) -> dict:
    """Read `link:` as an undirected graph and return connected components.

    Links are symmetric (declaring one side is enough) and transitive, and they do
    not depend on alias scores, so editing the dictionary never changes which parts
    light up together.

    Returns {pid: [pid, ...rest of its group]}.
    """
    order = {pid: i for i, pid in enumerate(parts_dict)}
    adj = {pid: set() for pid in parts_dict}
    for pid, spec in parts_dict.items():
        for other in (spec.get("link") or []):
            if other not in adj:
                print(f"[!] {pid}.link 의 '{other}' 는 없는 부위 — 무시")
                continue
            adj[pid].add(other)
            adj[other].add(pid)

    groups, seen = {}, set()
    for pid in parts_dict:
        if pid in seen:
            continue
        comp, stack = [], [pid]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x); comp.append(x)
            stack.extend(adj[x] - seen)
        comp.sort(key=lambda p: order[p])
        for x in comp:
            groups[x] = [x] + [y for y in comp if y != x]
    return groups


def shared_aliases(parts_dict: dict, groups: dict | None = None):
    """Aliases owned by several parts that no link ties together."""
    owner = {}
    for pid, spec in parts_dict.items():
        for al in (spec.get("aliases") or []):
            owner.setdefault(normalize(al), []).append(pid)
    out = []
    for al, pids in owner.items():
        if len(pids) < 2:
            continue
        g = set(groups.get(pids[0], [pids[0]])) if groups else set()
        if not all(p in g for p in pids):
            out.append((al, pids))
    return out


def score_all(text: str, parts: list[Part]):
    """[(pid, evidence, best_sim, pos)], best first."""
    q_norm = normalize(text)
    q_jamo = to_jamo(q_norm)
    ranked = []
    for p in parts:
        bs, ev, pos = p.eval(q_norm, q_jamo)
        ranked.append((p.pid, ev, bs, pos))
    ranked.sort(key=lambda x: (-x[1], x[3], -x[2]))
    return ranked


def best_match(text: str, parts: list[Part], tau: float = 70.0):
    """Top part above threshold, or None."""
    ranked = score_all(text, parts)
    if ranked and ranked[0][2] >= tau and ranked[0][1] > 0:
        return ranked[0][0], ranked[0][2]
    return None


class Matcher:
    """Stateful matcher with hysteresis to stop the target flickering.

    current_all is what actually gets drawn, driven by `link:` in parts.yaml.
    tie_group=True also groups parts whose alias scores tie, which is convenient
    but shifts whenever the dictionary changes.
    """
    def __init__(self, parts: list[Part], links: dict | None = None,
                 tau=70.0, margin=1.0, cooldown=0.8,
                 max_targets=3, tie_group=False):
        self.parts = parts
        self.links = links or {}
        self.tau, self.margin, self.cooldown = tau, margin, cooldown
        self.max_targets, self.tie_group = max_targets, tie_group
        self.current: str | None = None
        self.current_all: list[str] = []
        self._cur_score = 0.0
        self._last_switch = 0.0

    def _expand(self, ranked, pid):
        """Primary target -> everything that should light up with it."""
        if pid is None:
            return []
        out = list(self.links.get(pid, [pid]))
        if self.tie_group:
            for p in self._group(ranked, pid):
                if p not in out:
                    out.append(p)
        return out[:self.max_targets]

    def _group(self, ranked, pid):
        """Parts whose evidence ties with pid, in rank order."""
        key = next((k for p, k, *_ in ranked if p == pid), None)
        if not key:
            return [pid]
        grp = [p for p, k, *_ in ranked if k == key]
        if pid in grp:
            grp.remove(pid); grp.insert(0, pid)
        return grp[:self.max_targets]

    def update(self, text: str, now: float | None = None) -> str | None:
        now = time.monotonic() if now is None else now
        ranked = score_all(text, self.parts)
        if not ranked or ranked[0][2] < self.tau or ranked[0][1] <= 0:
            return self.current
        top_pid, top_key = ranked[0][0], ranked[0][1]
        if top_pid == self.current:
            self._cur_score = top_key
            self.current_all = self._expand(ranked, top_pid)
            return self.current
        cur_key = next((k for pid, k, *_ in ranked if pid == self.current), 0.0)
        if self.current is None or (top_key >= cur_key + self.margin and (now - self._last_switch) >= self.cooldown):
            self.current, self._cur_score, self._last_switch = top_pid, top_key, now
        self.current_all = self._expand(ranked, self.current)
        return self.current
