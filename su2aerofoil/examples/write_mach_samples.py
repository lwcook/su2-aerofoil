import json

import numpy as np

def main():
    N = 500
    alpha = 2
    beta = 2
    usamples = [-1. + 2.*np.random.beta(alpha, beta) for _ in np.arange(N)]

    with open('data/Mach_beta_samples.txt', 'w') as f:
        f.write(json.dumps(usamples))

if __name__ == "__main__":
    main()
