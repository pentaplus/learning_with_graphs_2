EXPLICIT FEATURE SPACES FOR LEARNING WITH GRAPHS

In this project, several graph kernels are evaluated in terms of
classification accuracy and runtime. Specifically, one implicit and five
explicit embedding methods are assessed. The random walk kernel is the implicit
embedding, while the explicit methods comprise the Weisfeiler-Lehman subtree
kernel, the neighborhood hash kernel (in three variants), the graphlet kernel
(in two variants), the shortest path kernel and the Eigen graph kernel. The
respective implementations can be found in the folder "code/embeddings".

The main module is "code/eval_embeddings.py". In this module, the embeddings
can be specified that are to be evaluated. Furthermore, the datasets can be
determined, on which the embeddings will run. See the comments within
"code/eval_embeddings.py" for more information.

The classification accuracies and runtimes are evaluated on the following eight
datasets: MUTAG, PTC(MR), ENZYMES, DD, NCI1, NCI109, FLASH CFG, and ANDROID
FCG. The datasets are contained in the folder "datasets".

The results of the experiments presented in the master's thesis can be found in
the folder "results_final".