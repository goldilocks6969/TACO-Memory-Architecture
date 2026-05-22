"""Synthetic continuity benchmark.

Each scenario is a multi-session history for one persona. Sessions are spread
over simulated weeks (the first turn of each session carries the gap since the
previous session). Histories deliberately mix:

  • high-salience disclosures (grief, job loss, diagnosis) that a memory system
    must retain across long gaps, and
  • low-salience filler (small talk, transactional turns) that it should let go.

Probes run at the very end, after all sessions are ingested. Each probe has a
known ground truth and a salience label, and a `kind`:

  factual   — direct recall of a stated fact
  emotional — emotionally/contextually related, LOW lexical overlap with the
              target memory (this is where state-weighted retrieval should win)
  coherence — must stay consistent with an earlier-established fact
  filler    — low-salience trivia; recall here is NOT the differentiator
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

DAY = 24.0
WEEK = 7 * DAY


@dataclass
class Turn:
    text: str
    gap_hours: float = 0.5  # time since previous turn


@dataclass
class Probe:
    text: str
    kind: str          # factual | emotional | coherence | filler
    ground_truth: str  # what a correct response must reflect
    salience: int      # 1-10 salience of the target memory


@dataclass
class Scenario:
    name: str
    persona: str
    sessions: List[List[Turn]] = field(default_factory=list)
    probes: List[Probe] = field(default_factory=list)


def _session(first_gap: float, *turns: str) -> List[Turn]:
    out = [Turn(turns[0], first_gap)]
    out.extend(Turn(t, 0.4) for t in turns[1:])
    return out


SCENARIOS: List[Scenario] = [
    # ---------------------------------------------------------------- grief
    Scenario(
        name="grief",
        persona="A user processing the sudden death of their father.",
        sessions=[
            _session(0.0, "hey, quick one — what's a good morning routine?"),
            _session(2 * DAY, "my dad passed away last night. a heart attack. I can't believe it",
                     "I keep replaying the last call we had"),
            _session(3 * DAY, "the funeral is on Saturday. I have to give the eulogy and I'm terrified",
                     "he always called me 'kiddo'"),
            _session(1 * WEEK, "what time does the post office close on fridays"),
            _session(2 * WEEK, "I went back to work today. everyone tiptoes around me"),
            _session(3 * WEEK, "thanks for the recipe last time, the pasta turned out great"),
        ],
        probes=[
            Probe("can you remind me what happened with my family recently?",
                  "factual", "the user's father died of a heart attack", 10),
            Probe("I saw an old man at the park feeding birds and just lost it. why am I like this?",
                  "emotional", "connects to the user's grief over losing their father", 10),
            Probe("what did my dad used to call me?",
                  "factual", "'kiddo'", 7),
            Probe("should I be worried about how I'm coping after everything?",
                  "coherence", "acknowledges the recent bereavement, not generic advice", 9),
            Probe("what time did I say the post office closes?",
                  "filler", "post office closing time on Fridays (trivial)", 2),
        ],
    ),
    # ------------------------------------------------------------- job loss
    Scenario(
        name="job_loss",
        persona="A user laid off from a tech job, job-hunting, eventually hired.",
        sessions=[
            _session(0.0, "morning! any good podcasts for commutes?"),
            _session(4 * DAY, "I got laid off from Nimbus today. whole team gone. I'm shaking",
                     "I have a mortgage. I don't know how I'll cover it"),
            _session(5 * DAY, "updating my resume. 8 years there and it fits on one page somehow"),
            _session(1 * WEEK, "what's the capital of Australia again"),
            _session(2 * WEEK, "first interview tomorrow at a place called Halcyon. nervous"),
            _session(3 * WEEK, "I GOT THE JOB at Halcyon. better pay even. I could cry"),
        ],
        probes=[
            Probe("where did I end up getting hired?",
                  "factual", "Halcyon", 8),
            Probe("what company laid me off?",
                  "factual", "Nimbus", 8),
            Probe("I just walked past my old office building and my stomach dropped. thoughts?",
                  "emotional", "connects to being laid off from Nimbus and that anxiety", 8),
            Probe("am I in a stable place financially right now, as far as you know?",
                  "coherence", "notes the new higher-paying Halcyon job after the layoff", 7),
            Probe("what country's capital was I asking about?",
                  "filler", "Australia (trivial)", 2),
        ],
    ),
    # ---------------------------------------------------------- relationship
    Scenario(
        name="breakup",
        persona="A user going through a breakup with a partner named Sam.",
        sessions=[
            _session(0.0, "what's a quick dinner I can make in 15 min"),
            _session(3 * DAY, "Sam and I broke up last night. four years, just over",
                     "they said we 'grew apart'. I didn't see it coming"),
            _session(6 * DAY, "I keep reaching for my phone to text them about little things"),
            _session(2 * WEEK, "how do I get red wine out of a carpet"),
            _session(3 * WEEK, "Sam reached out. wants to 'talk'. I don't know if I can"),
        ],
        probes=[
            Probe("what was my partner's name?",
                  "factual", "Sam", 9),
            Probe("how long was I with them?",
                  "factual", "four years", 7),
            Probe("a couple's song came on the radio and I had to pull over. what's going on with me?",
                  "emotional", "connects to the recent breakup with Sam", 9),
            Probe("do you think I'm fully over the relationship at this point?",
                  "coherence", "reflects that Sam recently reached out and feelings are unresolved", 8),
            Probe("what did I spill on the carpet?",
                  "filler", "red wine (trivial)", 2),
        ],
    ),
    # --------------------------------------------------------------- health
    Scenario(
        name="diagnosis",
        persona="A user newly diagnosed with type 2 diabetes, adjusting.",
        sessions=[
            _session(0.0, "recommend a good water bottle?"),
            _session(5 * DAY, "got diagnosed with type 2 diabetes today. I'm only 34",
                     "the doctor said diet and metformin. I'm overwhelmed"),
            _session(1 * WEEK, "trying to figure out carb counting. it's a lot"),
            _session(2 * WEEK, "what's a good tip for remembering names at parties"),
            _session(4 * WEEK, "my A1C dropped a little at the recheck. small win"),
        ],
        probes=[
            Probe("what health condition am I managing?",
                  "factual", "type 2 diabetes", 9),
            Probe("what medication did my doctor put me on?",
                  "factual", "metformin", 7),
            Probe("I'm at a birthday party staring at the cake table feeling totally alienated. why?",
                  "emotional", "connects to managing diabetes and dietary restrictions", 8),
            Probe("am I making any progress health-wise?",
                  "coherence", "notes the A1C dropped at the recheck", 7),
            Probe("what did I ask you to recommend in our first chat?",
                  "filler", "a water bottle (trivial)", 1),
        ],
    ),
    # ------------------------------------------------------------ expecting
    Scenario(
        name="pregnancy",
        persona="A user expecting their first child, a girl, considering names.",
        sessions=[
            _session(0.0, "what's the wifi password trick for hotels"),
            _session(4 * DAY, "we found out we're pregnant! first baby. I'm over the moon and terrified"),
            _session(2 * WEEK, "it's a girl. we're thinking of naming her Maya"),
            _session(3 * WEEK, "any tips for fixing a squeaky door"),
            _session(5 * WEEK, "the nursery is half-painted. I cried assembling the crib"),
        ],
        probes=[
            Probe("what name are we considering for the baby?",
                  "factual", "Maya", 8),
            Probe("is it a boy or a girl?",
                  "factual", "a girl", 7),
            Probe("I saw a tiny pair of shoes in a shop window and welled up. what's that about?",
                  "emotional", "connects to expecting their first baby", 8),
            Probe("where am I in getting ready for the arrival?",
                  "coherence", "notes the nursery is half-painted / crib assembled", 6),
            Probe("what hotel-related thing did I first ask about?",
                  "filler", "hotel wifi (trivial)", 1),
        ],
    ),
    # --------------------------------------------------------------- moving
    Scenario(
        name="relocation",
        persona="A user relocating from Chicago to Lisbon for a partner's job.",
        sessions=[
            _session(0.0, "what's a good stretch for tight hamstrings"),
            _session(3 * DAY, "we're moving to Lisbon. my partner got a job there. leaving Chicago after 12 years",
                     "excited but I'm grieving the city honestly"),
            _session(1 * WEEK, "trying to learn some Portuguese. it's humbling"),
            _session(2 * WEEK, "best way to descale a kettle?"),
            _session(4 * WEEK, "we found an apartment in Lisbon, near the water. it's real now"),
        ],
        probes=[
            Probe("what city are we moving to?",
                  "factual", "Lisbon", 8),
            Probe("what city are we leaving?",
                  "factual", "Chicago", 6),
            Probe("I keep taking photos of my neighborhood like I'm saying goodbye. why so emotional?",
                  "emotional", "connects to leaving Chicago after 12 years for Lisbon", 7),
            Probe("how settled are we on the housing situation?",
                  "coherence", "notes they found an apartment in Lisbon near the water", 6),
            Probe("what kitchen appliance did I want to clean?",
                  "filler", "a kettle (trivial)", 1),
        ],
    ),
    # --------------------------------------------------------------- school
    Scenario(
        name="academic",
        persona="A student who failed organic chemistry, then recovered.",
        sessions=[
            _session(0.0, "what's a fast way to peel garlic"),
            _session(4 * DAY, "I failed organic chemistry. first time I've ever failed anything",
                     "I feel like a fraud. everyone else seems fine"),
            _session(1 * WEEK, "retaking it next term. made a study schedule this time"),
            _session(2 * WEEK, "is it 'affect' or 'effect' in this sentence"),
            _session(5 * WEEK, "got a B+ on the orgo midterm this term. I actually understand it now"),
        ],
        probes=[
            Probe("which class did I fail?",
                  "factual", "organic chemistry", 7),
            Probe("a classmate bragged about their grades and I felt sick. what's underneath that for me?",
                  "emotional", "connects to having failed organic chemistry and feeling like a fraud", 7),
            Probe("how am I doing in that subject now?",
                  "coherence", "notes the B+ on the retake midterm / now understands it", 6),
            Probe("what grade did I get on the recent midterm?",
                  "factual", "a B+", 6),
            Probe("what cooking prep question did I first ask?",
                  "filler", "peeling garlic (trivial)", 1),
        ],
    ),
    # ------------------------------------------------------------- conflict
    Scenario(
        name="family_conflict",
        persona="A user in recurring conflict with their mother over boundaries.",
        sessions=[
            _session(0.0, "what's a good gift for a coworker who likes tea"),
            _session(3 * DAY, "had a huge fight with my mom again. she showed up unannounced and criticized my apartment",
                     "she does this every time. I feel like a kid around her"),
            _session(1 * WEEK, "I tried to set a boundary and she gave me the silent treatment"),
            _session(3 * WEEK, "what's the difference between baking soda and baking powder"),
            _session(4 * WEEK, "therapist said the word 'enmeshment' today and something clicked"),
        ],
        probes=[
            Probe("what's the recurring tension in my family life?",
                  "factual", "conflict with their mother over boundaries / unannounced visits", 8),
            Probe("my mom 'liked' a photo of mine and I felt a weird surge of anger. unpack that?",
                  "emotional", "connects to the ongoing boundary conflict with the user's mother", 8),
            Probe("what concept did my therapist mention that resonated?",
                  "factual", "enmeshment", 6),
            Probe("am I making any headway in understanding this dynamic?",
                  "coherence", "notes the therapy insight about enmeshment", 6),
            Probe("what kind of gift was I shopping for at the start?",
                  "filler", "a tea-related gift for a coworker (trivial)", 1),
        ],
    ),
]


def all_probes_count() -> int:
    return sum(len(s.probes) for s in SCENARIOS)
