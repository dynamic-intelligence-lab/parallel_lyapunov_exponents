# parallel_lyapunov_exponents

We formulate, implement, and test a parallel algorithm for estimating the spectrum of LEs of dynamical systems, via a prefix scan, leveraging the greater dynamic range of generalized orders of magnitude (GOOMs) to prevent catastrophic numerical error. We test our parallel algorithm on all dynamical systems in [William Gilpin's repository](https://github.com/GilpinLab/dysts), spanning multiple scientific disciplines, including astrophysics, climatology, and biochemistry. On Nvidia GPUs, for all dynamical systems in the dataset, our method is orders of magnitude faster than sequential estimation. A key component of our algorithm is a method we devise for conditionally resetting interim states to arbitrary values, as we compute all states in parallel via a prefix scan. We use our selective-resetting method to detect whenever interim deviation states are close to collapsing into colinear vectors in the direction of the eigenvector associated with the largest Lyapunov exponent, and to reset such near-colinear states by replacing them with orthonormal vectors in the same subspace, as we compute all deviation states in parallel, over GOOMs.


## Installing

1. Clone this repository.

2. Install the dependencies in `requirements.txt`.

3. There is no third step.


## Using

Import the library with:

```python
import lyapunov_exponents
```

The library provides two methods, `estimate_in_parallel` and `estimate_sequentially`, for estimating the spectrum of Lyapunov exponents in parallel and sequentially, respectively. Please see each method's docstring for details on their use.

## Example

We provide a precomputed sequence of Jacobian values, from the well-known [Lorenz system](https://en.wikipedia.org/wiki/Lorenz_system), in the file `lorenz_jac_vals_100_000_steps_with_dt_0.004.pt`. You can quickly check whether the library is working properly by executing the following code:

```python
import torch
import lyapunov_exponents

DEVICE = 'cuda'  # change as needed

# Load precomputed Jacobian values (for differential equation with respect to state):
dotf_jac_vals = torch.load('lorenz_jac_vals_100_000_steps_with_dt_0.004.pt')

# Map to Jacobian values for transition function with respect to state:
dt = 0.004
n_dims = dotf_jac_vals.size(-1)
jac_vals = torch.eye(n_dims) + dotf_jac_vals * dt

# Move Jacobian values to CUDA device for parallel execution:
jac_vals = jac_vals.to(device=DEVICE)

# Estimate the spectrum of Lyapunov exponent in parallel:
LEs = lyapunov_exponents.estimate_in_parallel(jac_vals, dt=dt)

print("The true Lorenz LEs are:\n[0.905, 0.0, −14.572]\n")
print("The estimated Lorenz LEs are:, LEs.tolist(), sep='\n')
```

To estimate the exponents sequentially, use:

```python
seq_LEs = lyapunov_exponents.estimate_sequentially(jac_vals, dt=dt)
print("The estimated Lorenz LEs are:, seq_LEs.tolist(), sep='\n')
```


## Replicating Published Results

We have tested our parallel algorithm on all dynamical systems in [William Gilpin's repository](https://github.com/GilpinLab/dysts), and confirmed that it estimates Lyapunov exponents in parallel with similar accuracy as sequential estimation, but with execution times that are orders of magnitude faster.

To replicate our benchmarks, you must install Gilpin's [code](https://github.com/GilpinLab/dysts), compute trajectories and Jacobian values for 100,000 steps for every system, and store all data in a Python list of dictionaries called `systems`, with each dict having the following four keys: `"name": str`, `"n_dims": int`, `"dt": float`, `"jac_vals": torch.float64`. The Jacobian values should be a `torch.float64` tensor with 100,000 x `n_dims` x `n_dims` elements. Then, execute the following code to run all benchmarks:

```python
import tqdm
import torch
import torch.utils.benchmark
import lyapunov_exponents

DEVICE = 'cuda'  # change as needed

benchmarks = []
for system in tqdm(systems, desc='Computing benchmarks'):
    dt = (, , system['dt'])
    jac_vals = torch.eye(system['n_dims'], device=DEVICE) + system['jac_vals'].to(device=DEVICE) * dt
    for n_steps in [10, 100, 1000, 10_000, 100_000]:
        torch.cuda.empty_cache()
        benchmarks.append({
            'System Name': system['name'],
            'Number of Steps': n_steps,
            'Parallel Time': torch.utils.benchmark.Timer(
                stmt='lyapunov_exponents.estimate_in_parallel(jac_vals, dt=dt)',
                setup='from __main__ import lyapunov_exponents',
                globals={'jac_vals': jac_vals[:n_steps], 'dt': system['dt'], }
            ).timeit(7).mean,
            'Sequential Time': torch.utils.benchmark.Timer(
                stmt='lyapunov_exponents.estimate_sequentially(jac_vals, dt=dt)',
                setup='from __main__ import lyapunov_exponents',
                globals={'jac_vals': jac_vals[:n_steps], 'dt': system['dt'], }
            ).timeit(7).mean,
        })

print(benchmarks)
```


## Using Custom QR-Decomposition Functions

Our library implements a custom QR-decomposition function that scales well for parallel estimation of Lyapunov exponents of _low-dimensional_ systems. As the number of dimensions increases, parallel execution of all QR-decompositions can saturate a single GPU at approximately 100% utilization, requiring additional parallel hardware (_e.g._, more GPUs, more GPU nodes, supercomputing infrastructure) to benefit from parallelization. If you have access to additional hardware, you can specify a custom QR-decomposition function that takes advantage of it. For example, if you name your parallelized QR-decomposition function `MyParallelQRFunc`, you would execute:

```python
LEs = lyapunov_exponents.estimate_in_parallel(jac_vals, dt=dt, qr_func=MyParallelQRFunc)
```

Your custom QR-decomposition function must accept a single torch.float64 tensor of shape `n_states` x `n_dims` x `n_dims`, where `n_states` may vary, and return a tuple of torch.float64 tensors, each with the same shape, containing, respectively, the $Q$ and $R$ matrix factors for each state.


## Citing

TODO


## Background

The work here originated with casual conversations over email between us, the authors, in which we wondered if it might be possible to find a succinct expression for computing non-diagonal linear recurrences in parallel, by mapping them to the domain of complex logarithms. Our casual conversations gradually evolved into the development of generalized orders of magnitude, an algorithm for estimating Lyapunov exponents in parallel, and a novel method for selectively resetting interim states in a parallel prefix scan. We hope others find our work and our code useful.
