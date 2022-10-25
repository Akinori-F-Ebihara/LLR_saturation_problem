# LLR saturation problem

This is an official repository of the paper, _Toward Asymptotic Optimality: Sequential Unsupervised Regression of Density Ratio for Early Classification_. In the paper, we defined and solved the _log likelihood ratio (LLR) saturation problem_ that hamper the Sequential Probability Ratio Test (SPRT) from reaching the theoretically-proven asymptotic optimality. Tensorflow implementations of the two proposed models, B2Bsqrt-TANDEM and TANDEformer, are found in the repo. We also list the detailed hyperparameter search space that is used to generate the experimental results.  

## Introduction
sequential density ratio estimation (SDRE)  

## Tested Environment
- Python 3.8
- Tensorflow 2.8.0
- CUDA 11.6.1
- cuDNN 8.3.3.40

## Code  
The models can readily be replaced with a conventional SDRE algorithm.  

## Hyperparameter Search Space
Attempt | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | #11
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---
Seconds | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 | 269
Seconds | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 | 269

| Attempt | #1    | #2    |
| :---:   | :---: | :---: |
| Seconds | 301   | 283   |

% \begin{table}[htbp]
% \centering
% % \small
% \caption{Hyperparameter search space. Note that the head size and number of attention heads are Transformer-specific.}
%   \begin{tabular}{cc}
%   Hyperparameter & Search space \\
%   \midrule
%   Weight decay &  $\{0.0, 10^{-5}, 10^{-4}, 10^{-3}\}$\\
%   Learning rate & $\{10^{-4}, 10^{-3}, 10^{-2}\}$  \\
%   Dropout & $\{0.0, 0.1, 0.2, 0.3, 0.4\}$  \\
%   Optimizer & $\{$Adam, RMSprop$\}$   \\
%   LLRe loss ratio & $\{$0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0$\}$   \\
%   Head size & $\{8, 16, 32\}$  \\
%   Num. heads ($n$)& $\{1, 2, 3\}$  \\
%   \bottomrule
%   \end{tabular}%
% \label{tab:hyperparameters}%
% \end{table}%