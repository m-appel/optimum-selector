# optimum-selector

“Optimize” a matrix by successively removing row-column pairs.

For now, this process only works on square matrices. Each iteration removes a
row-column pair as specified by a given score function. The score function takes
the current matrix as input and needs to return the next row-column index that
should be removed.

In addition, a summary function can be specified, which takes the whole matrix
as input and computes some value (e.g., mean of all cells).

The selector uses masked arrays, i.e., no data is actually removed. Therefore,
the indices remain stable during processing as the array shape stays the same.

The main script (`optimum_selector.py`) contains some example score functions.
For example, `min_mean_col` removes the row-column pair with the smallest mean
column value.

The `Selector` class expects a `numpy.ndarray` as input. The main script
currently only supports loading gzip-compressed CSV data that was written with
NumPy‘s `savetxt` function.

## Usage

The input needs to be a square 2D matrix supplied in the CSV format with a
header row. The header row should start with `#` since the data is loaded with
NumPy‘s `loadtxt` function. The headers are used to label the removed
row-column pair in addition to the index.

### Example input

```
# 123, 345, 567
   0 ,  1 ,  2
   3 ,  0 ,  4
   5 ,  6 ,  0
```

### Score Function

There currently is no elegant way to select the score function. The desired
function needs to be specified in the `Selector` constructor:
```python
selector = Selector(headers,
                    data,
                    score=max_weighted_dist, <- Here
                    summary=np.mean,
                    mask_value=args.mask_value)
```

### Example Call

```
python3 input_data.csv.gz output_data.csv.gz -m 0
```
The output is written to a gzip-compressed CSV file. The `-m` parameter can be
used to mask (ignore) an arbitrary value (`0` in this example) from the input
data and all calculations.

## Output Format

The output file contains three columns `summary,label,idx`, where `summary` is
the output of the specified summary function (if applicable) for the current
matrix, `label` is the label of the removed row-column pair, and `idx` is the
index.

The first row contains the summary value for the input matrix, i.e., before
anything is removed.
