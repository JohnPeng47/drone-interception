Dear Prof. [LASTNAME],
I'm a [YOUR CURRENT STATUS — e.g., final-year undergrad in X at Y] applying to [PROGRAM] for [TERM]. I came across your work on [SPECIFIC PAPER OR PROJECT] while looking into [SPECIFIC TECHNICAL AREA], and I'd like to ask whether you're considering taking master's students for [TERM].

The problem that have been recently preoccupying my attention is drone-to-drone interception as a counter-UAS solution for dismounted infantry. I think the solution here is a drone swarm that both operates as both detection and interception.

... [some reasons why] ...

Since this is quite a big problem, for my masters, I want to just focus on the interception piece. And here, I think there are two phases that should be completed sequentially:

Phase 1: Treat the problem as a traditional [Proportional Navigation Guidance] problem and establish a performance baseline. This is best formulated by the Beihang papers, which uses a vision derived LOS for guidance-based navigation, a DKF for the delay in the image processing stack, and an analytically derived function for calculating the lambda angle (from PN).
I also observe that under certain conditions, their control formulation collapses exactly into the classical PN/APN equation.

... [expand on Beihang papers] ...

Phase 2: Swap out PNG for a learned policy using RL
...
[
    - reproduce Gavin's but instead:
        a) use image/LOC as state observation
        b) modify the problem to intercept instead of catch
    - RL at the control level?
    - Combining this with noisy simulations of stereo observations from two observer drones
]
...

The problem I keep returning to is when learned policies should replace or augment classical homing guidance for interception.

The problem I keep returning to is when learned policies should replace or augment classical homing guidance for drone-to-drone interception. Yan 2024 (IEEE TIE) demonstrated that image-based visual servoing, effectively leveraging monocular vision guidance on a 2D plane, combined with proportional navigation — essentially classical missile homing with a CNN seeker replacing IR — achieves 80% interception against a non-maneuvering target in real outdoor flight.

PNG's well-known limitation is that it assumes a non-maneuvering or constant-acceleration target; against actively evading or learning adversaries, the guarantees break down. Gavin et al. 2025 showed that competitive multi-agent PPO with co-evolved pursuer/evader produces robust policies in this regime, but their pursuer observes ground-truth opponent state from motion capture and they explicitly defer perception. Meanwhile the synthetic-data drone-detection literature has produced credible CNN detectors trained entirely on rendered imagery, but none of it has been deployed inside a closed-loop interception controller. There's a coherent research program in closing those gaps — and a substantive question about when learned guidance is actually warranted versus when classical PNG with a noise-robust filter is sufficient.

Your group's work on [SPECIFIC THING — e.g., "agile quadrotor control" / "sim-to-real for aerial robots" / "learned visual control"] is what made me think this would be a productive place to develop the idea — particularly [SPECIFIC PAPER AND ONE-SENTENCE WHY]. I'd value the chance to discuss whether something in this direction could fit your group's research program.

Happy to send the longer proposal sketch, or to set up a brief call if you'd prefer to talk through it. Thank you for your time.

Best,
[YOU]
