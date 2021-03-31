# Remove terminal words in the output of CoreNLP and reduce the level of each tree.
eg.

`python -m src.utils.preprocess --reduce_tree --reduce_level=4 --reduce_output=output_trees < parse_trees`

content of parse_trees
```
{"ROOT": ["SQ-1"], "SQ-1": ["VBZ-1", "NP-1", "FRAG-1", ".-1"], "NP-1": ["EX-1"], "FRAG-1": ["NP-2",
"WHNP-1", "NP-3"], "NP-2": ["DT-1", "NN-1"], "WHNP-1": ["WDT-1"], "NP-3": ["JJ-1", "NNS-1", "NN-2"]}
```
# Collect symbols from corpus
`python -m src.utils.preprocess < dataset_files > output_symbols`

content of `dataset_files`

eg.
```
syntax_data/QQPPos_small/train/src
syntax_data/QQPPos_small/train/ref
syntax_data/QQPPos_small/val/src
syntax_data/QQPPos_small/val/ref
syntax_data/QQPPos_small/test/src
syntax_data/QQPPos_small/test/ref
```
