# TODO List

* Combine Init and Setup functions
* Set defaults for helper + loglikelihood
* Re-factor and simplify parameters + priors [Done]
* Re-factor kernels
* Documentation Bugs (ARPHMM)
* Demos should be in Jupyter Notebook, not ipython script
* Add "fit/predict" functions
    * Check code is efficiently implemented

* Check HMM preconditioning for pi -> need to use expanded_pi mode


* Add TQDM progress bars

* Refactor init and setup for sampler classes
    * Sample/Step method should take observations as an input

## Target API:
Input "dimensions of the model" -> Returns Sampler we everything setup
Optional input:
    * prior
    * initial parameters
    * forward + backward messages
    * helper
    * observations

Add "Fit" and "Predict" functions
Input to "Fit" would be similar to evaluator, requires: number of steps steps, step_kwargs tqdm, etc., returns parameters
Input to "Predict" would be observations, parameters (?), type of prediction and kwargs for the method. Should have version for the latent variables and the observations (smoothed, filtered, predictive for both). Returns prediction and SD?

I think the "Predict" method is the most complicated right now.
