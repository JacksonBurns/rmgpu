The LLMs developing `rmgpu` have been instructed to just _assume_ that Chemprop models for predicting the required quantities already exist.
This subdir will actually implement them.

This _should not_ be merged into `rmgpu` long-term -- it is outside the scope.
For now though, it will live here, since it has all the dependencies we need anyway.
Don't merge this into the LLM's `dev` branch -- their job is confusing enough already.

Notes for model development:
 - need to train with explicit hydrogens
 - need to use RIGR
 - some quantities are definitionally positive, bounded, etc.
    - add an extra output activation on chemprop fnn, not via normal output_transform because it is not applied during train, but could subclass! 
   - also need to move the log part of it into the transform, not preprocessing?

TODO:
 - how to handle different A units - even needed at all? everything (almost, only a few not) in rmgdb is in cm, mol, and s