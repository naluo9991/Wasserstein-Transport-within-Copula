# Wasserstein Transport between Copulas

Python/PyTorch implementation accompanying the manuscript:

**Na Luo and Oliver Grothe. _Wasserstein Transport between Copulas via a Benamou–Brenier-Type Dynamical Formulation._** (under review)

## Overview

This project studies how to continuously transform one copula into another while preserving uniform marginal distributions throughout the interpolation. It uses a constrained Benamou–Brenier formulation to approximate optimal transport paths, the associated velocity fields, and the copula-constrained Wasserstein distance.

A Sklar-type decomposition for flows reformulates the problem in a general density space. Neural networks parameterize an auxiliary density path and its dynamics, from which the copula interpolation is recovered. The framework also extends to distributions with arbitrary fixed continuous marginals; the repository includes an example with Gaussian marginals.

## Repository contents

| File or directory | Description |
| --- | --- |
| [`copula_2dim.ipynb`](copula_2dim.ipynb) | Two-dimensional copula interpolation, sample transport, and numerical evaluation. |
| [`copula_3dim.ipynb`](copula_3dim.ipynb) | Three-dimensional Gaussian copula interpolation and sample transport. |
| [`normal_fixed.ipynb`](normal_fixed.ipynb) | Transport between bivariate distributions with fixed Gaussian marginals. |
| [`result_path/`](result_path/) | Saved model weights for the numerical examples. |
| [`Fig/`](Fig/) | Figures showing densities, velocity fields, transported samples, and marginal histograms. |

The opening cells of each notebook define the models and shared computational routines. Subsequent experiment blocks apply the same workflow to different endpoint distributions.

## Numerical method

The implementation follows four main steps:

1. **Define the endpoints.** Specify the initial and target copulas and construct corresponding joint densities.
2. **Parameterize the flow.** Neural networks represent a correction to an interpolated log-density, an auxiliary velocity field, and a time-dependent normalization term. The density parameterization incorporates the prescribed endpoints.
3. **Optimize the transport.** Minimize a loss combining the transport energy of the recovered copula flow with a penalty on the continuity-equation residual.
4. **Recover and evaluate the copula path.** Compute marginal densities and cumulative distribution functions, recover the copula density and velocity field through the Sklar-type decomposition, and integrate the resulting ODE to transport samples.

Uniform marginals are imposed through the copula transformation. Marginal histograms and rank-based dependence measures provide numerical checks of the transported samples. The resulting paths and distances are numerical approximations.

See Section 4 and Appendix E of the manuscript for the numerical formulation and implementation details.

## Requirements

The notebooks use Python with NumPy, SciPy, Matplotlib, PyTorch, torchdiffeq, statsmodels, scikit-learn, and POT (`import ot`). Jupyter Notebook or JupyterLab is needed to run them interactively.

The code selects CUDA when available and otherwise uses the CPU. A CUDA-capable GPU is recommended for the full experiments.

## Usage

Clone the repository and start Jupyter from its root directory:

```bash
git clone https://github.com/naluo9991/Wasserstein-Transport-within-Copula.git
cd Wasserstein-Transport-within-Copula
jupyter notebook
```

Start with `copula_2dim.ipynb`. Run the setup and function/class definition cells, then select an experiment block.

**Using saved weights:** skip the training and model-saving cells, define the matching `target0` and `target1`, and load the corresponding checkpoint. The existing loading cells use filenames relative to the notebook directory. Point them to `result_path/`, for example:

```python
workspace = Workspace(args, target0, target1)
module = workspace.get_model()
module.load_state_dict(
    torch.load(
        "result_path/gauss9_reg5_point20_30000iter.pth",
        map_location=device,
    )
)
module.eval()
```

This checkpoint corresponds to the two-dimensional Gaussian copula example with correlations −0.9 and 0.9. Run the following plotting and evaluation cells for that experiment.

Alternatively, copy the selected checkpoint from `result_path/` into the repository root to use the existing relative filename.

**Training from scratch:** set the endpoint distributions and training parameters, then run the selected block containing `workspace.train()`. Retrieve the trained network with `module = workspace.get_model()` for subsequent evaluation. Running all cells also executes training for the later experiment blocks.

## Examples and results

The two-dimensional examples use the following endpoint pairs:

| Initial copula | Target copula |
| --- | --- |
| Gaussian, correlation −0.9 | Gaussian, correlation 0.9 |
| Gumbel, parameter 6 | Clayton, parameter 4 |
| Gaussian, correlation 0.8 | Clayton, parameter 2 |

The fixed-Gaussian-marginal notebook uses the same copula pairs. The three-dimensional notebook considers Gaussian copulas with different correlation matrices.

Outputs include interpolated densities, velocity fields, transported samples, marginal histograms, and transport-cost estimates. Saved illustrations are available in [`Fig/`](Fig/), including the [Gaussian copula density path](Fig/pdf_copula_gau9.pdf) and [transported samples](Fig/particle_gau9_copula.pdf).
