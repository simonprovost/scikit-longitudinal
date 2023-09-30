<!--suppress HtmlDeprecatedAttribute -->
<div align="center">
   <p align="center">
   <h1 align="center">
      <br>
      <a href="./../logo.png"><img src="./../logo.png" alt="Scikit-longitudinal" width="200"></a>
      <br>
      Scikit-longitudinal Experiments
      <br>
   </h1>
   <h4 align="center">Experimentation & Validation for Scikit-longitudinal Algorithms</h4>
</div>

> ⚠️ **DISCLAIMER**: This README focuses on the experiments conducted within the Scikit-longitudinal library. It is
> crucial to ensure the validity and reliability of the algorithms introduced. By running these experiments, we verify
> that our implementations align with the results presented in the corresponding research papers and that they achieve
> expected outcomes. Each file in this directory represents the various experiments conducted for different algorithms
> or tools.

## 🧪 Experiments

The following list showcases the primary experiments we've conducted, each corresponding to a specific algorithm or
tool:

### 📊 Algorithm Experiments

|                  Experiment Name                  |                     File Name & Link                     |                                                                    Description                                                                     |
|:-------------------------------------------------:|:--------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------:|
| **Correlation Based Feature Selection Per Group** |        [CFS Experiments.py](./CFS_Experiments.py)        | Implementation been validated by Authors: ⏳ TODO: As of today, this experiment needs to be re-done considering we re structured the CFS Per Group. |
|         **Lexicographical Random Forest**         |   [Lexico_rf_experiment.py](./Lexico_rf_experiment.py)   |                                                    Implementation been validated by Authors: ⏳                                                    |
|                 **Nested Trees**                  | [Nested_Tree_experiment.py](./Nested_Tree_experiment.py) |                                                    Implementation been validated by Authors: ⏳                                                     |

### 🛠️ Utility & Engine Tools

|       Utility Name       |     File Name & Link     |                                            Description                                             |
|:------------------------:|:------------------------:|:--------------------------------------------------------------------------------------------------:|
| **Experiment Utilities** |  [utils.py](./utils.py)  |                   Provides essential tools and helpers for running experiments.                    |
|  **Experiment Engine**   | [engine.py](./engine.py) | The core engine for the experiments. Offers comparative methods, window differentiators, and more. |

---

Remember, the goal of these experiments is not just to validate the algorithms, but also to ensure that our
reimplementations are accurate, reliable, and robust across various datasets and scenarios. They form a critical part of
the iterative development process, allowing us to refine, improve, and trust the tools we provide in the
Scikit-longitudinal library.
