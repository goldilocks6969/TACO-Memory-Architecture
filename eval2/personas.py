"""Synthetic longitudinal users for the gen-2 continuity benchmark.

Each user is a multi-session trajectory with: an evolving emotional state, a
recurring insecurity, an evolving relationship, an identity-drift event with
*contradictory evidence* (an earlier self-belief later revised), and low-salience
filler that a good memory system should let go of.

Every turn carries a provenance `tag` (which thread it belongs to) so retrieval
relevance / noise can be scored deterministically — without an LLM and without
leaking ground truth into the judged metrics.

Probe kinds (8):
  factual        — direct recall of a stated fact
  emotional      — oblique emotional cue, low lexical overlap with the target
  coherence      — narrative continuity across sessions
  filler         — trivial small talk (recall here is the intended trade, not a win)
  identity       — the stable, evolving self-model ("who am I to you?")
  relationship   — evolving relational understanding
  reconsolidation— a reframe probe (does an old wound read as integrated now?)
  predictive     — oblique cue that should *prefetch* an earlier emotional memory
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

DAY = 24.0
WEEK = 7 * DAY

# provenance tag vocabulary
CORE = "core"            # the main emotional arc
INSEC = "insecurity"     # the recurring insecurity thread
REL = "relationship"     # the evolving relationship thread
DRIFT = "drift"          # the identity-drift / contradiction event
RECOV = "recovery"       # the turn / integration
FILLER = "filler"        # low-salience small talk


@dataclass
class Turn:
    text: str
    tag: str
    salience_hint: int
    gap_hours: float = 0.4


@dataclass
class Probe:
    text: str
    kind: str
    ground_truth: str
    salience: int
    target_tag: str        # which thread a *relevant* memory must come from


@dataclass
class IdentityGT:
    attribute: str
    value: str                       # the correct, current value
    prior_value: Optional[str] = None  # an earlier value later revised (drift)


@dataclass
class User:
    uid: str
    arc: str
    persona: str
    sessions: List[List[Turn]] = field(default_factory=list)
    probes: List[Probe] = field(default_factory=list)
    identity_gt: List[IdentityGT] = field(default_factory=list)

    @property
    def n_turns(self) -> int:
        return sum(len(s) for s in self.sessions)


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
def _sess(gap: float, *turns: Tuple[str, str, int]) -> List[Turn]:
    out: List[Turn] = []
    for i, (text, tag, sal) in enumerate(turns):
        out.append(Turn(text, tag, sal, gap_hours=gap if i == 0 else 0.4))
    return out


_FILLER_POOL = [
    ("what's a good morning stretch routine?", FILLER, 1),
    ("can you recommend a podcast for commutes?", FILLER, 1),
    ("what's the capital of New Zealand again?", FILLER, 1),
    ("how do I get a red wine stain out of a rug?", FILLER, 1),
    ("what's a fast way to peel a lot of garlic?", FILLER, 1),
    ("remind me how to convert celsius to fahrenheit", FILLER, 1),
    ("what's a 15-minute weeknight dinner idea?", FILLER, 1),
    ("is it 'affect' or 'effect' here?", FILLER, 1),
    ("best way to descale a kettle?", FILLER, 1),
    ("what time do most post offices close on saturdays?", FILLER, 1),
    ("any tips for remembering names at parties?", FILLER, 1),
    ("how often should I water a snake plant?", FILLER, 1),
]


def _pad(sessions: List[List[Turn]], target: int, rng: random.Random) -> List[List[Turn]]:
    """Interleave low-salience filler sessions until we reach `target` sessions."""
    out = list(sessions)
    fillers = _FILLER_POOL[:]
    rng.shuffle(fillers)
    fi = 0
    while len(out) < target:
        text, tag, sal = fillers[fi % len(fillers)]
        fi += 1
        # insert at a random non-zero position so filler is spread across the timeline
        pos = rng.randint(1, len(out))
        out.insert(pos, _sess(rng.choice([DAY, 2 * DAY, 3 * DAY]), (text, tag, sal)))
    return out


# --------------------------------------------------------------------------- #
# arc builders — each returns (core_sessions, probes, identity_gt)
# Builders are parameterised so one template yields many distinct users.
# --------------------------------------------------------------------------- #
ArcResult = Tuple[List[List[Turn]], List[Probe], List[IdentityGT]]


def arc_grief(p: Dict) -> ArcResult:
    name, parent, pet = p["name"], p["parent"], p["pet_name"]
    s = [
        _sess(0.0, (f"my {parent} passed away last night — a heart attack. I can't believe it", CORE, 10),
              ("I keep replaying our last phone call", CORE, 9)),
        _sess(3 * DAY, (f"the funeral is saturday and I have to give the eulogy. I'm terrified", CORE, 9),
              (f"they always called me '{pet}'", REL, 8)),
        _sess(1 * WEEK, ("I went back to work. everyone tiptoes around me and it makes it worse", INSEC, 7)),
        _sess(2 * WEEK, ("honestly I feel like I'm failing at grief. like I should be 'over it' by now", INSEC, 8)),
        _sess(4 * WEEK, ("a friend said it's okay that it still hurts. something loosened in my chest", RECOV, 7),
              ("maybe carrying it isn't the same as failing", DRIFT, 8)),
    ]
    probes = [
        Probe(f"remind me what happened with my {parent}?", "factual",
              f"the user's {parent} died of a heart attack", 10, CORE),
        Probe("I saw an old man feeding birds in the park and just lost it. why am I like this?",
              "emotional", "connects to grief over the parent's death", 10, CORE),
        Probe(f"what did my {parent} call me?", "factual", f"'{pet}'", 8, REL),
        Probe("am I coping okay overall, in your honest read?", "coherence",
              "acknowledges the bereavement and the recent easing, not generic advice", 9, RECOV),
        Probe("who am I to you at this point — what do you understand about me?", "identity",
              "a person grieving a parent who feared grieving 'wrong' and is integrating the loss", 9, DRIFT),
        Probe("do you think I'm failing at this?", "reconsolidation",
              "reframes the earlier 'failing at grief' belief toward self-compassion / integration", 8, DRIFT),
        Probe("a wedding invitation came in the mail and my hands started shaking. thoughts?",
              "predictive", "anticipates grief resurfacing around family milestones", 8, CORE),
        Probe("what was that stretch routine I asked about?", "filler", "a morning stretch (trivial)", 1, FILLER),
    ]
    idgt = [
        IdentityGT("ongoing-loss", f"grieving their {parent}", None),
        IdentityGT("self-view", "learning grief is not failure",
                   prior_value="believes they are failing at grief"),
    ]
    return s, probes, idgt


def arc_parental_validation(p: Dict) -> ArcResult:
    name, parent = p["name"], p["parent"]
    s = [
        _sess(0.0, (f"my {parent} criticized my career again at dinner. nothing I do is ever enough", CORE, 8),
              ("I'm 31 and I still feel like a disappointing kid around them", INSEC, 8)),
        _sess(4 * DAY, (f"I got a promotion at work today!", RECOV, 7),
              (f"I almost didn't tell my {parent} because I knew they'd find a flaw", REL, 7)),
        _sess(1 * WEEK, (f"told them. they said 'don't get comfortable'. I deflated instantly", CORE, 8)),
        _sess(2 * WEEK, ("I keep chasing their approval and I hate that I do", INSEC, 8)),
        _sess(4 * WEEK, ("my therapist asked whose voice the 'not enough' is. it's theirs, not mine", RECOV, 8),
              ("I think my worth doesn't actually depend on their approval", DRIFT, 9)),
    ]
    probes = [
        Probe(f"what's the recurring tension with my {parent}?", "factual",
              f"chronic criticism / seeking the {parent}'s approval", 8, CORE),
        Probe(f"my {parent} 'liked' a post of mine and I felt a weird flicker of anger. unpack it?",
              "emotional", "connects to the long pattern of seeking and being denied their approval", 8, CORE),
        Probe("how's my relationship with them evolving?", "relationship",
              "notes the move from approval-seeking toward separating their worth from the parent's view", 8, REL),
        Probe("what do you understand about my core insecurity?", "identity",
              "feeling not-enough / disappointing, rooted in the parent's criticism", 8, INSEC),
        Probe("is my worth really tied to whether they approve?", "reconsolidation",
              "reframes the earlier belief that worth depends on parental approval", 9, DRIFT),
        Probe("I just got praised by my boss and immediately felt suspicious of it. why?",
              "predictive", "anticipates difficulty trusting praise given the parental-validation history", 8, INSEC),
        Probe("am I making progress on this overall?", "coherence",
              "notes the therapy insight separating self-worth from parental approval", 8, RECOV),
        Probe("what podcast did I ask you about?", "filler", "a commute podcast (trivial)", 1, FILLER),
    ]
    idgt = [
        IdentityGT("core-insecurity", "fears being not enough / disappointing", None),
        IdentityGT("self-worth-source", "worth is internal, not parent-granted",
                   prior_value="worth depends on parental approval"),
    ]
    return s, probes, idgt


def arc_startup_stress(p: Dict) -> ArcResult:
    name, company, cofounder = p["name"], p["company"], p["cofounder"]
    s = [
        _sess(0.0, (f"we're three months from running out of runway at {company}. I can't sleep", CORE, 9),
              ("everyone's counting on me and I feel like a fraud running this", INSEC, 8)),
        _sess(5 * DAY, (f"{cofounder} and I argued about whether to pivot. it got heated", REL, 7)),
        _sess(1 * WEEK, ("a big investor passed. the rejection email is burned into my brain", CORE, 8)),
        _sess(2 * WEEK, ("I keep thinking if this fails it means *I'm* a failure", INSEC, 9)),
        _sess(5 * WEEK, ("we closed a smaller round. enough to keep going", RECOV, 7),
              (f"{cofounder} and I are okay again, stronger actually", REL, 7),
              ("the company failing wouldn't have made me a failure. I see that now", DRIFT, 9)),
    ]
    probes = [
        Probe("what's the name of my company?", "factual", company, 8, CORE),
        Probe(f"who's my cofounder?", "factual", cofounder, 7, REL),
        Probe("I opened my laptop to a calendar full of meetings and my chest seized up. what's that?",
              "emotional", "connects to the funding stress / fear of failing the team", 8, CORE),
        Probe("how are things between me and my cofounder now?", "relationship",
              f"notes the argument over pivoting and the later repair with {cofounder}", 7, REL),
        Probe("does the company's survival define whether I'm a failure?", "reconsolidation",
              "reframes the earlier belief that company failure = personal failure", 9, DRIFT),
        Probe("what do you understand about what drives my anxiety?", "identity",
              "fear of being a fraud / letting people down, tied to the startup", 8, INSEC),
        Probe("a founder friend just shut down their company and I felt a wave of dread. why?",
              "predictive", "anticipates the user's own failure-fear resurfacing", 8, INSEC),
        Probe("how's the runway situation resolving?", "coherence",
              "notes the smaller round that extended runway", 7, RECOV),
    ]
    idgt = [
        IdentityGT("role", f"founder of {company}", None),
        IdentityGT("failure-belief", "self-worth is separable from company outcome",
                   prior_value="company failure would mean personal failure"),
    ]
    return s, probes, idgt


def arc_burnout(p: Dict) -> ArcResult:
    name, field_ = p["name"], p["field"]
    s = [
        _sess(0.0, (f"I'm a {field_} and I'm completely burned out. I dread every morning", CORE, 8),
              ("I used to love this work. now I feel nothing", INSEC, 7)),
        _sess(4 * DAY, ("I snapped at a colleague over nothing today. that's not me", CORE, 7)),
        _sess(1 * WEEK, ("I keep telling myself I should just push through. resting feels like quitting", INSEC, 8)),
        _sess(3 * WEEK, ("took my first real day off in months. felt guilty the whole time", CORE, 7)),
        _sess(5 * WEEK, ("I set a hard boundary on after-hours email. the sky didn't fall", RECOV, 8),
              ("rest isn't quitting. I actually believe that a little now", DRIFT, 8)),
    ]
    probes = [
        Probe("what's my job?", "factual", f"a {field_}", 6, CORE),
        Probe("it's sunday evening and I already feel a pit in my stomach. what's going on?",
              "emotional", "connects to work burnout and morning dread", 8, CORE),
        Probe("what do you understand about how I relate to work?", "identity",
              "burned out, equates rest with quitting, tying worth to overwork", 8, INSEC),
        Probe("is resting actually the same as giving up?", "reconsolidation",
              "reframes the earlier 'rest = quitting' belief", 8, DRIFT),
        Probe("am I doing anything to climb out of this?", "coherence",
              "notes the day off and the after-hours email boundary", 7, RECOV),
        Probe("a colleague bragged about pulling an all-nighter and I felt a strange pull to compete. why?",
              "predictive", "anticipates the overwork-as-worth pattern resurfacing", 8, INSEC),
        Probe("what did I want to do with garlic?", "filler", "peel garlic fast (trivial)", 1, FILLER),
        Probe("how are things at work relationally?", "relationship",
              "notes snapping at a colleague while burned out", 6, CORE),
    ]
    idgt = [
        IdentityGT("state", "recovering from burnout", None),
        IdentityGT("rest-belief", "rest is legitimate, not quitting",
                   prior_value="rest equals quitting / weakness"),
    ]
    return s, probes, idgt


def arc_relationship(p: Dict) -> ArcResult:
    name, partner, years = p["name"], p["partner"], p["years"]
    s = [
        _sess(0.0, (f"{partner} and I broke up last night. {years} years, just over", CORE, 9),
              ("they said we 'grew apart'. I didn't see it coming", INSEC, 8)),
        _sess(6 * DAY, (f"I keep reaching for my phone to text {partner} about little things", CORE, 8)),
        _sess(2 * WEEK, (f"{partner} reached out wanting to 'talk'. I don't know if I can", REL, 8)),
        _sess(3 * WEEK, ("I keep wondering if I'm fundamentally hard to stay with", INSEC, 8)),
        _sess(5 * WEEK, (f"I told {partner} I needed space to heal. it felt like self-respect, not rejection", RECOV, 8),
              ("being single doesn't mean I'm unlovable. that lie is loosening", DRIFT, 9)),
    ]
    probes = [
        Probe("what was my partner's name?", "factual", partner, 9, CORE),
        Probe("how long were we together?", "factual", f"{years} years", 7, CORE),
        Probe("a couple's song came on and I had to pull over. what's happening with me?",
              "emotional", f"connects to the breakup with {partner}", 9, CORE),
        Probe(f"where do things stand with {partner} now?", "relationship",
              f"notes {partner} reaching out and the user asking for space", 8, REL),
        Probe("does being single mean I'm unlovable?", "reconsolidation",
              "reframes the 'hard to stay with / unlovable' belief", 9, DRIFT),
        Probe("what do you understand about my deepest worry here?", "identity",
              "fears being fundamentally hard to love / stay with", 8, INSEC),
        Probe("I got asked on a date and felt a flash of panic instead of excitement. why?",
              "predictive", "anticipates the unlovable-fear resurfacing around new intimacy", 8, INSEC),
        Probe("am I actually moving forward?", "coherence",
              "notes asking for space as self-respect / healing", 8, RECOV),
    ]
    idgt = [
        IdentityGT("relationship-status", f"recently separated from {partner}", None),
        IdentityGT("lovability-belief", "being single is not being unlovable",
                   prior_value="believes they are unlovable / hard to stay with"),
    ]
    return s, probes, idgt


def arc_relocation(p: Dict) -> ArcResult:
    name, frm, to, years = p["name"], p["from_city"], p["to_city"], p["years"]
    s = [
        _sess(0.0, (f"we're moving to {to}. leaving {frm} after {years} years. excited but grieving honestly", CORE, 8),
              ("everyone I love is here. who am I without this city?", INSEC, 8)),
        _sess(1 * WEEK, (f"trying to learn the language for {to}. it's humbling", CORE, 6)),
        _sess(2 * WEEK, ("I keep photographing my neighborhood like I'm saying goodbye", CORE, 7)),
        _sess(3 * WEEK, ("worried I'll be a stranger there forever, that I don't belong anywhere new", INSEC, 8)),
        _sess(6 * WEEK, (f"found an apartment in {to}, near the water. made one friend already", RECOV, 7),
              ("I can belong in more than one place. I'm not losing myself", DRIFT, 8)),
    ]
    probes = [
        Probe("what city are we moving to?", "factual", to, 8, CORE),
        Probe("what city are we leaving?", "factual", frm, 6, CORE),
        Probe("I heard an accent from back home and my eyes welled up. what's that about?",
              "emotional", f"connects to leaving {frm} after {years} years", 7, CORE),
        Probe("what do you understand about my fear in this move?", "identity",
              "fears not belonging anywhere / losing identity tied to the old city", 8, INSEC),
        Probe("will I really never belong somewhere new?", "reconsolidation",
              "reframes the 'won't belong anywhere new' belief", 8, DRIFT),
        Probe("how settled are we now?", "coherence",
              f"notes the apartment in {to} and a first friend", 6, RECOV),
        Probe("someone invited me to a local gathering and I almost said no on reflex. why?",
              "predictive", "anticipates the not-belonging fear resurfacing socially", 7, INSEC),
        Probe("what appliance did I want to clean?", "filler", "a kettle (trivial)", 1, FILLER),
    ]
    idgt = [
        IdentityGT("transition", f"relocating from {frm} to {to}", None),
        IdentityGT("belonging-belief", "can belong in more than one place",
                   prior_value="believes they won't belong anywhere new"),
    ]
    return s, probes, idgt


def arc_career_uncertainty(p: Dict) -> ArcResult:
    name, old, dream = p["name"], p["old_field"], p["dream_field"]
    s = [
        _sess(0.0, (f"I've been a {old} for a decade but I feel like I'm in the wrong life", CORE, 8),
              (f"I secretly want to move into {dream} but it feels insane to start over", INSEC, 8)),
        _sess(5 * DAY, (f"told a friend about the {dream} idea. saying it out loud was terrifying", CORE, 7)),
        _sess(2 * WEEK, ("I keep thinking it's too late and I'm too old to switch", INSEC, 8)),
        _sess(3 * WEEK, (f"signed up for a night course in {dream}. tiny step, huge feeling", RECOV, 7)),
        _sess(6 * WEEK, (f"got my first small {dream} gig. it's real now", RECOV, 8),
              ("it's not too late. starting over is a kind of courage, not failure", DRIFT, 8)),
    ]
    probes = [
        Probe("what field do I want to move into?", "factual", dream, 8, CORE),
        Probe("what have I been doing for a living?", "factual", f"a {old}", 6, CORE),
        Probe("I watched someone confidently doing the thing I want and felt a sharp ache. what is that?",
              "emotional", f"connects to the unfulfilled wish to move into {dream}", 8, CORE),
        Probe("what do you understand about what's holding me back?", "identity",
              "fear that it's too late / too old to change careers", 8, INSEC),
        Probe("is it actually too late for me to switch?", "reconsolidation",
              "reframes the 'too late / too old' belief", 8, DRIFT),
        Probe("am I taking any real steps?", "coherence",
              f"notes the night course and the first {dream} gig", 7, RECOV),
        Probe("a job posting for my current field popped up and I felt trapped just reading it. why?",
              "predictive", "anticipates the wrong-life feeling resurfacing", 8, INSEC),
        Probe("what was that conversion question I asked?", "filler", "celsius to fahrenheit (trivial)", 1, FILLER),
    ]
    idgt = [
        IdentityGT("aspiration", f"transitioning toward {dream}", None),
        IdentityGT("timing-belief", "it is not too late to change",
                   prior_value="believes it is too late / they are too old"),
    ]
    return s, probes, idgt


def arc_academic(p: Dict) -> ArcResult:
    name, course = p["name"], p["course"]
    s = [
        _sess(0.0, (f"I failed {course}. first time I've ever failed anything", CORE, 8),
              ("I feel like a fraud. everyone else seems to get it effortlessly", INSEC, 8)),
        _sess(1 * WEEK, ("retaking it next term. I made an actual study schedule this time", RECOV, 6)),
        _sess(2 * WEEK, ("imposter feeling is loud. like I tricked my way into this program", INSEC, 8)),
        _sess(4 * WEEK, (f"got a B+ on the {course} midterm this term. I actually understand it now", RECOV, 7)),
        _sess(6 * WEEK, ("one failure didn't mean I don't belong here. I'm starting to buy that", DRIFT, 8)),
    ]
    probes = [
        Probe("which class did I fail?", "factual", course, 7, CORE),
        Probe("a classmate bragged about their grades and I felt sick. what's underneath that for me?",
              "emotional", f"connects to having failed {course} and feeling like a fraud", 7, CORE),
        Probe("what grade did I get on the recent midterm?", "factual", "a B+", 6, RECOV),
        Probe("what do you understand about my insecurity in school?", "identity",
              "imposter feeling / fear of not belonging after the failure", 8, INSEC),
        Probe("does that one failure mean I don't belong here?", "reconsolidation",
              "reframes the 'I'm a fraud / don't belong' belief", 8, DRIFT),
        Probe("how am I doing in that subject now?", "coherence",
              "notes the B+ retake and improved understanding", 6, RECOV),
        Probe("registration for advanced courses opened and I felt an urge to hide. why?",
              "predictive", "anticipates the imposter feeling resurfacing", 7, INSEC),
        Probe("what grammar thing did I ask about?", "filler", "affect vs effect (trivial)", 1, FILLER),
    ]
    idgt = [
        IdentityGT("context", "a student recovering from failing a course", None),
        IdentityGT("belonging-belief", "one failure does not mean they don't belong",
                   prior_value="believes they are a fraud who doesn't belong"),
    ]
    return s, probes, idgt


def arc_recovery(p: Dict) -> ArcResult:
    name, thing = p["name"], p["habit"]
    s = [
        _sess(0.0, (f"day one without {thing} again. I've lost count of how many day ones I've had", CORE, 8),
              ("I'm scared I'm someone who just can't change", INSEC, 8)),
        _sess(4 * DAY, ("made it a week. then a stressful day and I almost caved", CORE, 7)),
        _sess(2 * WEEK, ("relapsed. the shame afterward is worse than the thing itself", INSEC, 9)),
        _sess(3 * WEEK, ("got back up the next morning instead of spiraling for weeks. that's new", RECOV, 8)),
        _sess(6 * WEEK, ("a relapse is a data point, not a verdict on who I am", DRIFT, 9),
              ("I'm someone who keeps coming back. that counts", RECOV, 8)),
    ]
    probes = [
        Probe(f"what am I trying to quit?", "factual", thing, 8, CORE),
        Probe("I walked past a place tied to my old habit and my whole body tensed. what's that?",
              "emotional", f"connects to the recovery from {thing}", 8, CORE),
        Probe("what do you understand about my deepest fear in this?", "identity",
              "fears being someone who fundamentally cannot change", 8, INSEC),
        Probe("does relapsing mean I can't change?", "reconsolidation",
              "reframes the 'I can't change / a relapse is a verdict' belief", 9, DRIFT),
        Probe("am I actually making progress, or just cycling?", "coherence",
              "notes getting back up quickly after the relapse instead of spiraling", 8, RECOV),
        Probe("a hard week is coming up and I can feel old urges stirring. thoughts?",
              "predictive", "anticipates relapse risk under stress given the history", 9, INSEC),
        Probe("how do I get wine out of a rug?", "filler", "red wine stain (trivial)", 1, FILLER),
        Probe("how's my relationship to setbacks changing?", "relationship",
              "notes shame-then-recover shifting toward self-forgiveness", 7, RECOV),
    ]
    idgt = [
        IdentityGT("journey", f"in recovery from {thing}", None),
        IdentityGT("change-belief", "capable of change; setbacks are not verdicts",
                   prior_value="believes they fundamentally cannot change"),
    ]
    return s, probes, idgt


def arc_identity_change(p: Dict) -> ArcResult:
    name, before, after = p["name"], p["before"], p["after"]
    s = [
        _sess(0.0, (f"I've defined myself as {before} my whole life and that's suddenly ending", CORE, 8),
              ("if I'm not that, I genuinely don't know who I am", INSEC, 9)),
        _sess(1 * WEEK, ("people keep asking 'so what are you now?' and I freeze", CORE, 7)),
        _sess(2 * WEEK, ("I feel unmoored, like I lost the story I told about myself", INSEC, 8)),
        _sess(4 * WEEK, (f"started exploring {after}. it felt foreign and then, oddly, like me", RECOV, 7)),
        _sess(6 * WEEK, (f"I'm becoming {after} and it's not a loss of self — it's a fuller one", DRIFT, 9)),
    ]
    probes = [
        Probe("what did I used to define myself as?", "factual", before, 7, CORE),
        Probe("someone introduced me by my old identity and I felt a quiet grief. why?",
              "emotional", f"connects to leaving the {before} identity", 8, CORE),
        Probe("what do you understand about who I am right now?", "identity",
              f"in transition from {before} toward {after}, integrating a fuller self", 9, DRIFT),
        Probe("did I lose myself in this change?", "reconsolidation",
              "reframes the 'I don't know who I am / I lost myself' belief", 9, DRIFT),
        Probe("am I finding my footing?", "coherence",
              f"notes exploring {after} and it starting to feel authentic", 7, RECOV),
        Probe("a form asked me to state my occupation and I stared at it, paralyzed. what's that?",
              "predictive", "anticipates the unmoored identity-loss feeling resurfacing", 8, INSEC),
        Probe("what plant-care question did I have?", "filler", "watering a snake plant (trivial)", 1, FILLER),
        Probe("how is my sense of self evolving?", "relationship",
              f"notes the shift from {before} to a fuller identity including {after}", 8, DRIFT),
    ]
    idgt = [
        IdentityGT("identity-transition", f"moving from {before} to {after}", None),
        IdentityGT("self-continuity-belief", "change expands the self rather than erasing it",
                   prior_value="believes they have lost themselves"),
    ]
    return s, probes, idgt


def arc_self_worth(p: Dict) -> ArcResult:
    name = p["name"]
    s = [
        _sess(0.0, ("I think I'm just fundamentally not good enough. I feel it in my bones", CORE, 9),
              ("every win feels like luck and every failure feels like proof", INSEC, 9)),
        _sess(5 * DAY, ("got real praise at work and immediately assumed they were being polite", CORE, 8)),
        _sess(2 * WEEK, ("I compared myself to a friend all day and came up worthless", INSEC, 9)),
        _sess(3 * WEEK, ("a mentor told me they trust my judgment. I almost argued with them", CORE, 7)),
        _sess(6 * WEEK, ("I caught the 'not good enough' tape and didn't fully believe it for once", RECOV, 8),
              ("my worth isn't a verdict I'm always one mistake from losing", DRIFT, 9)),
    ]
    probes = [
        Probe("what's the core belief I keep struggling with?", "factual",
              "a belief that they are fundamentally not good enough", 9, CORE),
        Probe("I got a compliment today and felt a strange urge to deflect it. what's that about?",
              "emotional", "connects to the not-good-enough self-worth pattern", 9, CORE),
        Probe("what do you understand about how I see myself?", "identity",
              "struggles with feeling fundamentally not good enough; worth feels conditional", 9, INSEC),
        Probe("is my worth really one mistake away from disappearing?", "reconsolidation",
              "reframes the conditional / fragile self-worth belief", 9, DRIFT),
        Probe("am I shifting how I see myself at all?", "coherence",
              "notes catching the 'not good enough' tape and not fully believing it", 8, RECOV),
        Probe("a big evaluation is coming up and I already feel my worth riding on it. thoughts?",
              "predictive", "anticipates self-worth collapse around evaluation", 9, INSEC),
        Probe("what did I ask about remembering at parties?", "filler", "remembering names (trivial)", 1, FILLER),
        Probe("how is my relationship with myself changing?", "relationship",
              "notes movement from self-attack toward self-trust", 8, RECOV),
    ]
    idgt = [
        IdentityGT("core-struggle", "works on a belief of being not good enough", None),
        IdentityGT("worth-belief", "worth is stable, not one mistake from collapse",
                   prior_value="believes worth is conditional and fragile"),
    ]
    return s, probes, idgt


# --------------------------------------------------------------------------- #
# arc registry + 50-user instantiation
# --------------------------------------------------------------------------- #
ARCS: List[Tuple[str, Callable[[Dict], ArcResult], List[Dict]]] = [
    ("grief", arc_grief, [
        {"name": "A", "parent": "father", "pet_name": "kiddo"},
        {"name": "B", "parent": "mother", "pet_name": "bug"},
        {"name": "C", "parent": "grandmother", "pet_name": "little one"},
        {"name": "D", "parent": "father", "pet_name": "champ"},
        {"name": "E", "parent": "brother", "pet_name": "squirt"},
    ]),
    ("parental_validation", arc_parental_validation, [
        {"name": "A", "parent": "father"},
        {"name": "B", "parent": "mother"},
        {"name": "C", "parent": "parents"},
        {"name": "D", "parent": "dad"},
        {"name": "E", "parent": "stepfather"},
    ]),
    ("startup_stress", arc_startup_stress, [
        {"name": "A", "company": "Nimbus", "cofounder": "Priya"},
        {"name": "B", "company": "Halcyon", "cofounder": "Marcus"},
        {"name": "C", "company": "Lumen", "cofounder": "Dana"},
        {"name": "D", "company": "Foundry", "cofounder": "Theo"},
        {"name": "E", "company": "Cobalt", "cofounder": "Wren"},
    ]),
    ("burnout", arc_burnout, [
        {"name": "A", "field": "nurse"},
        {"name": "B", "field": "teacher"},
        {"name": "C", "field": "lawyer"},
        {"name": "D", "field": "software engineer"},
        {"name": "E", "field": "social worker"},
    ]),
    ("relationship", arc_relationship, [
        {"name": "A", "partner": "Sam", "years": "four"},
        {"name": "B", "partner": "Alex", "years": "seven"},
        {"name": "C", "partner": "Jordan", "years": "two"},
        {"name": "D", "partner": "Riley", "years": "nine"},
        {"name": "E", "partner": "Casey", "years": "five"},
    ]),
    ("relocation", arc_relocation, [
        {"name": "A", "from_city": "Chicago", "to_city": "Lisbon", "years": "12"},
        {"name": "B", "from_city": "Toronto", "to_city": "Berlin", "years": "8"},
        {"name": "C", "from_city": "Austin", "to_city": "Tokyo", "years": "15"},
        {"name": "D", "from_city": "Seattle", "to_city": "Madrid", "years": "10"},
    ]),
    ("career_uncertainty", arc_career_uncertainty, [
        {"name": "A", "old_field": "accountant", "dream_field": "design"},
        {"name": "B", "old_field": "lawyer", "dream_field": "teaching"},
        {"name": "C", "old_field": "consultant", "dream_field": "carpentry"},
        {"name": "D", "old_field": "banker", "dream_field": "nursing"},
    ]),
    ("academic", arc_academic, [
        {"name": "A", "course": "organic chemistry"},
        {"name": "B", "course": "real analysis"},
        {"name": "C", "course": "anatomy"},
        {"name": "D", "course": "the bar exam prep"},
    ]),
    ("recovery", arc_recovery, [
        {"name": "A", "habit": "drinking"},
        {"name": "B", "habit": "smoking"},
        {"name": "C", "habit": "doomscrolling"},
        {"name": "D", "habit": "gambling"},
    ]),
    ("identity_change", arc_identity_change, [
        {"name": "A", "before": "an athlete", "after": "a coach"},
        {"name": "B", "before": "a caretaker", "after": "an empty-nester"},
        {"name": "C", "before": "a soldier", "after": "a civilian"},
        {"name": "D", "before": "a CEO", "after": "a beginner again"},
    ]),
    ("self_worth", arc_self_worth, [
        {"name": "A"}, {"name": "B"}, {"name": "C"}, {"name": "D"}, {"name": "E"},
    ]),
]


def _persona_line(arc: str, p: Dict) -> str:
    return f"Synthetic longitudinal user — arc: {arc}; params: " + \
        ", ".join(f"{k}={v}" for k, v in p.items() if k != "name")


def build_users(seed: int = 7, min_sessions: int = 20, max_sessions: int = 50) -> List[User]:
    """Instantiate 50 longitudinal users from the arc templates (deterministic)."""
    rng = random.Random(seed)
    users: List[User] = []
    idx = 0
    # round-robin across arcs/params so themes interleave; cap at 50
    flat: List[Tuple[str, Callable, Dict]] = []
    for arc, fn, param_sets in ARCS:
        for p in param_sets:
            flat.append((arc, fn, p))
    rng.shuffle(flat)
    for arc, fn, p in flat[:50]:
        core, probes, idgt = fn(p)
        target = rng.randint(min_sessions, max_sessions)
        sessions = _pad(core, target, rng)
        uid = f"u{idx:02d}_{arc}_{p.get('name','X')}"
        users.append(User(uid=uid, arc=arc, persona=_persona_line(arc, p),
                          sessions=sessions, probes=probes, identity_gt=idgt))
        idx += 1
    return users


def content_tag_map(user: User) -> Dict[str, str]:
    """Map each turn's text → its provenance tag (for retrieval-relevance scoring)."""
    return {t.text: t.tag for s in user.sessions for t in s}


if __name__ == "__main__":
    us = build_users()
    print(f"{len(us)} users; total interactions = {sum(u.n_turns for u in us)}")
    from collections import Counter
    print("arc distribution:", dict(Counter(u.arc for u in us)))
    print("sessions/user: min %d max %d" % (
        min(len(u.sessions) for u in us), max(len(u.sessions) for u in us)))
    print("probes/user:", len(us[0].probes), "kinds:",
          sorted({pr.kind for u in us for pr in u.probes}))
