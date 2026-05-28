# Research

## Phase 1: Approaching Drone Detection using CV
### Goal
- Create a dataset for detecting drones from the perspective of a moving drone along an approaching flight path
- Use synthetic images using methodology laid out in papers
-> These images should be generated from multiple points on multiple interception trajectories
### Evaluation
- Evaluate on real drone pictures from the training set
- Evaluate on flight with two drones
### Outcome
- The outcome here is a synthetic data pipeline that can be used to feed into Phase 2
> Note: we are *not* going to re-use the trained model, but simply proving that the generated synthetic data can be used to effectively train a drone end to end

## Phase 2: Train an End to End Image -> Motor Interception RL Model
