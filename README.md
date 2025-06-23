# parallel_lyapunov_exponents

Initial reference implementation of the algorithm we propose for estimating the Lyapunov exponents of dynamical systems in parallel, with a prefix scan, leveraging generalized orders of magnitude (GOOMs) to be able to handle a larger dynamic range of magnitudes than would be possible with torch.float64. Example:

```python
import torch
import lyapunov_exponents

DEVICE = 'cuda'  # change as needed

system = torch.load('lorenz.pt')
jac_vals, dt, n_dims = (system['jac_vals'], system['dt'], system['n_dims'])
if system['is_continuous']:
    jac_vals = torch.eye(n_dims) + jac_vals * dt  # Euler approximation
jac_vals = jac_vals.to(device=DEVICE)

LEs = lyapunov_exponents.estimate_spectrum_in_parallel(jac_vals, dt=dt)
print(LEs.tolist())
```

## Installing

1. Clone this repository.

2. Install all dependencies in `requirements.txt`.

3. There is no third step.


## Using

Import the library with:

```python
import lyapunov_exponents
```

The library provides three public methods:

* `estimate_spectrum_in_parallel`, for estimating the spectrum of Lyapunov exponents in parallel, applying the parallel algorithm we propose, incorporating our selective-resetting method, as described in our paper;

* `estimate_largest_in_parallel`, for estimating only the largest Lyapunov exponent in parallel, applying the expression we algebraically derive in Appendix B of our paper; and

* `estimate_spectrum_sequentially`, for estimating the spectrum of Lyapunov exponents sequentially, applying the standard method with sequential QR-decompositions. We provide this method as a convenience, for benchmarking purposes.

Below we show how to use all methods to estimate Lyapunov exponents for one dynamical system. Please see each method's docstring for additional details on how to use it.


## Example

We provide a precomputed sequence with 100,000 Jacobian values from the well-known [Lorenz system](https://en.wikipedia.org/wiki/Lorenz_system) in the file `lorenz.pt`. You can quickly test that the library is working properly by executing the following code:

```python
import torch
import lyapunov_exponents

DEVICE = 'cuda'  # change as needed

# Load sample system data:
system = torch.load('lorenz.pt')
jac_vals, dt, n_dims = (system['jac_vals'], system['dt'], system['n_dims'])

# If necessary, map to Jacobian values of transition func with respect to state:
if system['is_continuous']:
    jac_vals =  torch.eye(n_dims) + jac_vals * dt  # Euler approximation

# Move Jacobian values to cuda device for parallel execution:
jac_vals = jac_vals.to(device=DEVICE)

# Estimate the spectrum of Lyapunov exponents in parallel:
LEs = lyapunov_exponents.estimate_spectrum_in_parallel(jac_vals, dt=dt)
print("Estimated spectrum of Lyapunov exponents for {}:".format(system['name']))
print(LEs.tolist())
```

For comparison, the true Lyapunov Exponents of Lorenz are estimated to be `[0.905, 0.0, −14.572]`.

To estimate only the largest Lyapunov exponents in parallel, use:

```python
LLE = lyapunov_exponents.estimate_largest_in_parallel(jac_vals, dt=dt)
print("Parallel estimated largest Lyapunov exponent for {}:\n{}".format(system['name'], LLE.item()))
```

To estimate the spectrum of Lyapunov exponents sequentially, use:

```python
seq_LEs = lyapunov_exponents.estimate_spectrum_sequentially(jac_vals, dt=dt)
print("Sequential estimated spectrum of Lyapunov exponents for {}:\n{}".format(system['name'], seq_LEs.tolist()))
```


## Replicating Published Results

We have tested our parallel algorithm on all dynamical systems modeled in [William Gilpin's repository](https://github.com/GilpinLab/dysts), and confirmed that our algorithm estimates the spectrum of Lyapunov exponents in parallel with comparable accuracy to sequential estimation, but with execution times that are orders of magnitude faster.

To replicate our benchmarks, please install Gilpin's [code](https://github.com/GilpinLab/dysts), compute a sequence of 100,000 Jacobian values for every system, and store the resulting data in a Python list of dictionaries called `systems`, with each dictionary having the following keys: `"name": str`, `"is_continuous": bool`, `"n_dims": int`, `"dt": float`, `"jac_vals": torch.float64`. The Jacobian values, `"jac_vals"`, should be in the form a `torch.float64` tensor with `100,000` x `n_dims` x `n_dims` elements.

Once you have computed the data for all systems, execute the following code to run all benchmarks:

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
            stmt='lyapunov_exponents.estimate_spectrum_in_parallel(jac_vals, dt=dt)',
            setup='from __main__ import lyapunov_exponents',
            globals={'jac_vals': jac_vals[:n_steps], 'dt': dt, }
        ).timeit(7).mean

        pbar.set_description("{}, {:,} steps, sequential, 7 runs".format(system['name'], n_steps))
        sequential_mean_time = torch.utils.benchmark.Timer(
                stmt='lyapunov_exponents.estimate_spectrum_sequentially(jac_vals, dt=dt)',
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


## Scaling Parallel Estimation of Spectrum to Higher-Dimensional Systems

Our library implements a custom QR-decomposition function that scales well for parallel estimation of the spectrum of Lyapunov exponents of _low-dimensional_ systems. As the number of dimensions increases, parallel execution of all QR-decompositions eventually saturates a single GPU at approximately 100% utilization, requiring additional parallel hardware (_e.g._, more GPUs, more GPU nodes, supercomputing infrastructure) to benefit from parallelization. If you have access to additional parallel hardware, you can specify a custom QR-decomposition function that takes advantage of it. For example, if your parallelized QR-decomposition function is called `MyParallelQRFunc`, you would execute:

```python
LEs = lyapunov_exponents.estimate_spectrum_in_parallel(jac_vals, dt=dt, qr_func=MyParallelQRFunc)
```

Your custom QR-decomposition function must accept a single torch.float64 tensor of shape `...` x `n_dims` x `n_dims`, where `...` can be any number of dimensions, and return a tuple of torch.float64 tensors, _each_ with the same shape (`...` x `n_dims` x `n_dims`), containing, respectively, the $Q$ and $R$ factors for each matrix.

Our parallel method for estimating only the largest Lyapunov exponent scales well to higher-dimensional systems.


## Citing

TODO: Update citation.

```
@misc{heinsenkozachkov2025gooms,
    title={
        Generalized Orders of Magnitude for
        Scalable, Parallel, High-Dynamic-Range Computation},
    author={Franz A. Heinsen, Leo Kozachkov},
    year={2025},
}
```

## Notes

The work here originated with casual conversations over email between us, the authors, in which we wondered if it might be possible to find a succinct expression for computing non-diagonal linear recurrences in parallel, by mapping them to the complex plane. Our casual conversations gradually evolved into the development of generalized orders of magnitude, along with an algorithm for estimating Lyapunov exponents in parallel, and a novel method for selectively resetting interim states in a parallel prefix scan.

We hope others find our work and our code useful.
