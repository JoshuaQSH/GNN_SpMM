## GCN_SparseMatrix
Test whether Different Sparse Matrix Format would have an imapct on GCN forward/backward process.

## GPU SpMM Test
Please see the `./cupy` for the CUDA SpMM by using cupy with python [On-going]. 
Should consider dgSPARSE for more speedup. 

## CPU SpMM and GCN Test
Plese see the `./cpu` for the CPU SpMM and GCN test.
A Colab .ipynb is offered for the easy evaluation.
See `./cpu/XGBoost_SpMM_analysis.ipynb` for more details, which included:
 - [] SpMM matrix generation
 - [] XGBoost building and fine-tuning
 - [] Other testing method including naive CNN, KNN, Decision Tree, SVM 
