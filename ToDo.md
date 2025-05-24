# ToDo

- [ ] Rewrite the Encoder/decoder pipeline pipelines.py... it's currently fugly vibe-code. Hopefully FAIR will release their new fairseq2 that supports torch2.7.0-cu128 soon
- [ ] make sure the contextualizer runs once, but all the 100/40 denoiser layer iterations run
- [ ] rename the contextualizer 
- [ ] make sure the contextualizer uses causal self-attention
- [ ] make sure the denoiser uses cross-attention
- [ ] make sure the AdaLN timestep modulation works in tandem with cross-attention
- [ ] make sure its AdamW
