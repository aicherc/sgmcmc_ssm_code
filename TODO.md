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
Focus on smoothed first, then filtered/predictive

Plan of attack
* [x] API for SGMCMCSampler + HELPER
* [x] Implement for HMM + LGSSM
* [ ] Implement for PFs
* [ ] Implement for SLDS
* [ ] Extend to SeqSGMCMCSampler
* [ ] Do implementation of sample version
* [ ] Online Evaluator Refactor?

Before Release
* [ ] Small Sample Data Set (Maybe Gaussian Ion Channel? Or ECG data?)
* [ ] Figures on real data + synthetic data

Sampler
    __init__()
    setup() (depreciated)
    noisy_logjoint(),
    noisy_loglikelihood(),
    noisy_gradient()
    fit()
        step_sgd, step_adagrad,
        sample_gibbs,
        sample_sgld, sample_sgrld,
        project_parameters
    predict()
        sample_latent,
        sample_y

    'predictive' stuff needs to be an option rather than a separate function

Helper
    __init__(forward_message, backward_message, **kwargs)
    message_passing:
        forward_message, forward_pass,
        backward_message, backward_pass,
    loglikelihood:
        marginal_loglikelihood, 
        complete_data_loglikelihood,
        pf_loglikelihood,
    predictive_loglikelihood:
        predictive_loglikelihood, 
        pf_predictive_loglikelihood,
    gradients:
        gradient_marginal_loglikelihood,
        gradient_complete_data_loglikelihood,
        pf_gradient_loglikelihood
    gibbs:
        calc_gibbs_sufficient_statistic,
        parameters_gibbs_sample,
    latent_var:
        latent_var_sample,
        latent_var_marginal
        pf_latent_var_marginal
    y:
        y_sample,
        y_marginal,

Should probably split latent_var_sample -> samples from marginal distribution and samples from joint
What should API be?

- For Analytic Message Passing
predict -> target = latent vs y
        -> kind = marginal vs joint vs pairwise
        -> lag (only for marginal would indicate how many)
        -> return_distr (boolean) (only for marginal or pairwise)
        -> num_samples None or >= 1 (only for marginal or joint)
simulate -> simulate data based on initial state?
         -> num_samples
         -> message pass over data to get initial state then run generate_data
refactor marginal FFBS? How to sample lag = k generically (2 cases)

- For PFs
predict -> target = latent vs y
    -> kind = marginal vs joint vs pairwise
    -> lag (for marginal)
    -> return_distr (must be true)
    -> N number of particles
simulate -> simulate data based on initial state?
         -> num_samples (is output)
         -> N is number of particles to approximate latent filtered distribution
         -> particle smoother to get particle cloud then sample + run generate_data

### REFACTOR OFFLINE
- Refactorization of OfflineEvaluator
-> Take Times as an optional argument -> adds it to get_metrics [x]
    -> or get_metrics + get_samples takes times as an optional argument? Done

-> evaluate_timed (function similar to fit_timed)
-> tqdm manual updates with tqdm.update()

- Refactorization of OnlineEvaluator
-> refactor evaluate/sample_step to match fit + fit_timed
-> maybe pass evaluator to online evaluator?
    -> Can call evaluate at appropriate times (?)

- Refactor plotting utils to allow x axis to be based on time [x] (for metrics)

Pass TQDM to noisy_loglikelihood / noisy_logjoint

fit/evaluate functions with try catch to output what they can (?) Low Priority



# 1/16
-> Finish Predict in SGMCMCSAMPLER
-> Debug SeqSampler
-> Figure Out why SVM doesn't like exchange rate data ...
