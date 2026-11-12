# Toward Scalable and Valid Conditional Independence Testing with Spectral Representations

[Paper (PDF)](https://raw.githubusercontent.com/mlresearch/v306/main/assets/frohlich26a/frohlich26a.pdf)

This repository contains the code necessary for reproducing the experiments of the paper "Toward Scalable and Valid Conditional Independence Testing with Spectral Representations" (ICML, 2026).

## Install instructions

Clone the repository and install dependencies with conda:

```bash
git clone https://github.com/alekfrohlich/SCIT.git
cd SCIT
conda env create -f environment.yaml
conda activate reproducing_scit
```

## Reproducing synthetic experiments

From the repository root, run 

```bash
python reproduce.py
```

This reproduces figures 2 and 5 of the paper using hyperparameter values as described in Appendix C. Experimental results are saved to `results/`. Note: reproduction scripts are being finalized.

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@InProceedings{frohlich2026spectralcit,
  author = {Fr\"{o}hlich, Alek and Kostic, Vladimir R and Lounici, Karim and Perazzo, Daniel and Tiezzi, Daniel Guimar\~{a}es and Pontil, Massimiliano},
  title = {Toward Scalable and Valid Conditional Independence Testing with Spectral Representations},
  booktitle = {International Conference on Machine Learning},
  year = {2026},
  month = {Jul},
}
```
