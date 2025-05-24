# Data-Driven Law Firm Rankings to Reduce Information Asymmetry in Legal Disputes

This repository contains the code for AHPI from our paper
**[Data-Driven Law Firm Rankings to Reduce Information Asymmetry in Legal Disputes](http://arxiv.org/abs/2408.16863)**.

We present a ranking algorithm **AHPI** which assigns scores to entities (e.g. law firms) competing against each other in pairwise interactions (e.g. trials). The pairwise interactions can be of different “types” (e.g. civil rights trials as opposed to torts trials)<sup>1</sup>  and include asymmetry (e.g. in a trial a defendant has a priori higher winning odds than the plaintiff).  
We assign strength scores to law firms based on historical outcomes. **AHPI** is based on a generalised Bradley-Terry model. For an exemplified application of **AHPI**, see the [law_firm_ranking repository](https://github.com/mojona/law_firm_ranking).


---

## Repository Overview

- `environment.yml`: Required dependencies. Create the environment with  
  ```bash
  conda env create -f environment.yml
  conda activate law_firm_ranking
  ```

- `AHPI.py`: Core implementation of the **AHPI algorithm**.

---

## Questions?

For questions, bug reports, or collaboration opportunities, please contact:
**slera@mit.edu**
