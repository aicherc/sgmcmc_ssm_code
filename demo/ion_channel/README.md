These python scripts demo fitting hidden Markov models with Gaussian emissions (GaussHMM) on the ion channel data `alamethicin.mat` from .

The scripts demonstrate how to fit GaussHMM using SGMCMC and Gibbs sampling.

* `ion_channel_subset_demo.py` - This script uses a downsampled subset of the ion channel data (T = 10,000, downsampled by factor of 50)
* `ion_channel_downsample_demo.py` - This script uses a downsampled version of the ion channel data (downsampled by factor of 50)
* `ion_channel_full_demo.py` - This script uses the full ion channel data. We do not compare with Gibbs sampling, which is too slow for this data.

Additional documentation can be found within each script.
