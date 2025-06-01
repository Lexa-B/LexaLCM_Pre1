# ToDo

## Critical

## High

- [ ] PreNet: Normalizes and centers the input embeddings
- [ ] PostNet: Denormalizes the output and shifts it back to the SONAR embedding space by the noralizer's original offset.

## Medium

- [ ] Make all causal masks bool. They're currently Float32.
- [ ] Exponential Moving Average (EMA) for weight stabilization during training or inference

## Low

- [ ] Rewrite the SONAR Encoder/decoder pipeline pipelines.py... it's currently fugly vibe-code. Hopefully FAIR will release their new fairseq2 that supports torch2.7.0-cu128 soon

## Lowest
=