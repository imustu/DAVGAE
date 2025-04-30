# DAVGAE

**Predicting disease-gene associations** 

## Package Requirements

```python
Python >=3.8
torch
torch_geometric
numpy
pandas
sklearn
```

## Data Source

[datasetA](https://snap.stanford.edu/biodata/datasets/10028/10028-PP-Miner.html) Reference: Szklarczyk, Damian, et al. "The STRING database in 2017: quality-controlled protein–protein association networks, made broadly accessible." _Nucleic Acids Research_. 2016.

[datasetB](https://github.com/yangkuoone/HerGePred/tree/master) Reference: K. Yang et al., "Hergepred: Heterogeneous network embedding representation for disease gene prediction." _IEEE Journal of Biomedical and Health Informatics_, vol. 23, no. 4, pp. 1805–1815, Jul. 2019.

[datasetC](https://github.com/juanshu30/Disease-Gene-Prioritization-with-Privileged-Information-and-Heteroscedastic-Dropout) Reference: J. Shu, Y. Li, S. Wang, B. Xi, and J. Ma, "Disease gene prediction with privileged information and heteroscedastic dropout." _Bioinformatics_, vol. 37, no. 1, pp. 410–417, Jul. 2021.

## To run DAVAGE

python main.py --dropout {dropout} --EPOCHS {epoch} --HIDDEN-SIZE {hidden_size} --OUT-CHANNELS {out_ch} --HEADS {head} --output-path test_out/her/{command_counter}/

## License

Yuan Zhang, Juan Wang, Jiajie Xing, Xiaomin Chen. Identifying disease-gene associations by topological and biological feature-based data augmentation and graph neural networks, IEEE Journal of Biomedical and Health Informatics, 2025.
