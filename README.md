# optimum-selector
“Optimize” a matrix by successively removing row-column pairs.

For now, this process only works on square matrices. Each iteration removes a
row-column pair as specified by a given score function. The score function takes
the current matrix as input and needs to return the next row-column index that
should be removed.

The selector uses masked arrays, i.e., no data is actually removed. Therefore,
the indices remain stable during processing as the array shape stays the same.

The main script (`optimum_selector.py`) contains some example score functions.
For example, `min_mean_col` removes the row-column pair with the smallest mean
column value.

The `Selector` class expects a `numpy.ndarray` as input. The main script
currently only supports loading gzip-compressed CSV data that was written with
NumPy‘s `savetxt` function.
