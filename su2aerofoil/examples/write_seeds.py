import json

import numpy as np

def main():
    N = 500
    xseeds = [np.random.uniform(0., 1.) for _ in np.arange(N)]

    with open('data/Mach_scaled_seeds.txt', 'w') as f:
        f.write(json.dumps(xseeds))

if __name__ == "__main__":
    main()
