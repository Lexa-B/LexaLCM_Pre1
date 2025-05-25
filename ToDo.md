# ToDo

## Critical

- [ ] make sure the contextualizer runs once, but all the 100/40 denoiser layer iterations run
- [ ] add a gaussian noise generator to the denoiser
- [ ] make sure the contextualizer uses causal self-attention
- [ ] make sure the denoiser uses cross-attention

## High

- [ ] make sure the AdaLN timestep modulation works in tandem with cross-attention

## Medium

- [ ] make sure its AdamW
- [ ] Exponential Moving Average (EMA) for weight stabilization during training or inference

## Low

- [ ] Rewrite the SONAR Encoder/decoder pipeline pipelines.py... it's currently fugly vibe-code. Hopefully FAIR will release their new fairseq2 that supports torch2.7.0-cu128 soon

## Lowest






