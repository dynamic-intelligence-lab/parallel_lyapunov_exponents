# parallel_lyapunov_exponents

Initial reference implementation of the algorithm we propose for estimating the spectrum of Lyapunov exponents of a dynamical system in parallel, with a prefix scan incorporating a novel selective-resetting mechanism. Our algorithm operates over generalized orders of magnitude (GOOMs) to be able to handle a larger dynamic range of magnitudes than is possible with torch.float64, and uses our selective-resetting method for detecting whenever interim deviation states are close to collapsing into colinear vectors in the direction of the eigenvector associated with the largest Lyapunov exponent, and for resetting such near-colinear states by replacing them with orthonormal vectors in the same subspace, as the prefix scan updates all deviation states in parallel, over GOOMs. We have tested the implementation in this repository on all dynamical systems from [William Gilpin's repository](https://github.com/GilpinLab/dysts), spanning multiple scientific disciplines, including astrophysics, climatology, and biochemistry. On a recent midtier Nvidia GPU, our method is orders of magnitude faster than sequential estimation for all systems in Gilpin's repository.


## Installing

1. Clone this repository.

2. Install the dependencies in `requirements.txt`.

3. There is no third step.


## Using

Import the library with:

```python
import lyapunov_exponents
```

The library provides two methods, `estimate_in_parallel` and `estimate_sequentially`, for estimating the spectrum of Lyapunov exponents in parallel and sequentially, respectively, of any dynamical system. Below, as an example, we show how to use both methods to one dynamical system. Please see each method's docstring for additional details on how to use them.


## Example

We provide a precomputed sequence of Jacobian values, from the well-known [Lorenz system](https://en.wikipedia.org/wiki/Lorenz_system), in the file `lorenz.pt`. You can quickly test that the library is working properly by executing the following code:

```python
import torch
import lyapunov_exponents

DEVICE = 'cuda'  # change as needed

# Load sample system data:
system = torch.load('lorenz.pt')
jac_vals, dt, n_dims = (system['jac_vals'], system['dt'], system['n_dims'])

# If necessary, map to Jacobian values of transition func with respect to state
if system['is_continuous']:
    jac_vals =  torch.eye(n_dims) + jac_vals * dt  # Euler approximation

# Move Jacobian values to cuda device for parallel execution:
jac_vals = jac_vals.to(device=DEVICE)

# Estimate the spectrum of Lyapunov exponents in parallel:
LEs = lyapunov_exponents.estimate_in_parallel(jac_vals, dt=dt)
print("The estimated Lyapunov Exponents for {} are:\n{}".format(system['name'], LEs.tolist()))
```

We also provide a method for estimating the spectrum of exponents sequentially:

```python
seq_LEs = lyapunov_exponents.estimate_sequentially(jac_vals, dt=dt)
print("The estimated Lorenz LEs are:, seq_LEs.tolist(), sep='\n')
```

For comparison, the true Lyapunov Exponents of Lorenz are estimated to be `[0.905, 0.0, −14.572]`.



## Estimating the Largest Lyapunov Exponent in Parallel

TODO: Add write-up and code for estimating largest Lyapunov exponent.


## Replicating Published Results

We have tested our parallel algorithm on all dynamical systems modeled in [William Gilpin's repository](https://github.com/GilpinLab/dysts), and confirmed that our algorithm estimates the spectrum of Lyapunov exponents in parallel with similar accuracy as sequential estimation, but with execution times that are orders of magnitude faster.

To replicate our benchmarks, you must install Gilpin's [code](https://github.com/GilpinLab/dysts), compute a sequence of 100,000 Jacobian values for every system, and store all data in a Python list of dictionaries called `systems`, with each dictionary having the following keys: `"name": str`, `"is_continuous": bool`, `"n_dims": int`, `"dt": float`, `"jac_vals": torch.float64`. The Jacobian values should be in the form a `torch.float64` tensor with `100,000` x `n_dims` x `n_dims` elements.

Once you have the data ready, execute the following code to run all benchmarks:

```python
import torch
import torch.utils.benchmark
import lyapunov_exponents
from tqdm import tqdm

DEVICE = 'cuda'  # change as needed

benchmarks = []
pbar = tqdm(systems)  # iterator with progress bar

for system in pbar:

    jac_vals, dt, n_dims = (system['jac_vals'], system['dt'], system['n_dims'])
    if system['is_continuous']:
        jac_vals =  torch.eye(n_dims) + jac_vals * dt  # Euler approximation
    jac_vals = jac_vals.to(device=DEVICE)

    for n_steps in [10, 100, 1000, 10_000, 100_000]:

        pbar.set_description("{}, {:,} steps, parallel, 7 runs".format(system['name'], n_steps))
        parallel_mean_time = torch.utils.benchmark.Timer(
            stmt='lyapunov_exponents.estimate_in_parallel(jac_vals, dt=dt)',
            setup='from __main__ import lyapunov_exponents',
            globals={'jac_vals': jac_vals[:n_steps], 'dt': dt, }
        ).timeit(7).mean

        pbar.set_description("{}, {:,} steps, sequential, 7 runs".format(system['name'], n_steps))
        sequential_mean_time = torch.utils.benchmark.Timer(
                stmt='lyapunov_exponents.estimate_sequentially(jac_vals, dt=dt)',
                setup='from __main__ import lyapunov_exponents',
                globals={'jac_vals': jac_vals[:n_steps], 'dt': dt, }
            ).timeit(7).mean

        benchmarks.append({
            'System Name': system['name'],
            'Number of Steps': n_steps,
            'Parallel Time (Mean of 7 Runs)': parallel_mean_time,
            'Sequential Time (Mean of 7 Runs)': sequential_mean_time,
        })

torch.save(benchmarks, 'benchmarks.pt')
print(*benchmarks, sep='\n')
```


## Scaling to Greater Number of Dimensions with Custom QR-Decomposition Functions

Our library implements a custom QR-decomposition function that scales well for parallel estimation of Lyapunov exponents of _low-dimensional_ systems. As the number of dimensions increases, parallel execution of all QR-decompositions can saturate a single GPU at approximately 100% utilization, requiring additional parallel hardware (_e.g._, more GPUs, more GPU nodes, supercomputing infrastructure) to benefit from parallelization. If you have access to additional hardware, you can specify a custom QR-decomposition function that takes advantage of it. For example, if you name your parallelized QR-decomposition function `MyParallelQRFunc`, you would execute:

```python
LEs = lyapunov_exponents.estimate_in_parallel(jac_vals, dt=dt, qr_func=MyParallelQRFunc)
```

Your custom QR-decomposition function must accept a single torch.float64 tensor of shape `n_states` x `n_dims` x `n_dims`, where `n_states` may vary, and return a tuple of torch.float64 tensors, each with the same shape, containing, respectively, the $Q$ and $R$ matrix factors for each state.


## Citing

TODO


## Background

The work here originated with casual conversations over email between us, the authors, in which we wondered if it might be possible to find a succinct expression for computing non-diagonal linear recurrences in parallel, by mapping them to the domain of complex logarithms. Our casual conversations gradually evolved into the development of generalized orders of magnitude, an algorithm for estimating Lyapunov exponents in parallel, and a novel method for selectively resetting interim states in a parallel prefix scan. We hope others find our work and our code useful.
