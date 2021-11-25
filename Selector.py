import time
from typing import Any, Callable, List

import numpy as np
import numpy.ma as ma

class Selector:
    """Optimize a data matrix by removing row-column pairs the order
    specified by a given score function.

    Parameters
    ----------
    headers: list of str
        List of column labels
    data: np.ndarray
        Input data matrix with shape n x n
    score: callable
        Score function that takes a matrix as input and returns the
        row-column index that should be removed.
    summary: callable, optional
        An optional summary function that is called after every
        iteration. Takes a matrix as input.
    break_condition: callable, optional
        An optional break condition that is evaluated after every
        iteration. Takes a matrix as input and returns a boolean.
    target_count: int, default=1
        Stop iteration when `target_count` row-column pairs are left.

    Notes
    -----
    `score` function signature:
        score(np.ndarray) -> int
    `summary` function signature:
        summary(np.ndarray) -> Any
    `break_condition` function signature:
        break_condition(np.ndarray) -> bool
    """
    def __init__(self,
                 headers: List[Any],
                 data: np.ndarray,
                 score: Callable[[np.ndarray], int],
                 summary: Callable[[np.ndarray], Any] = None,
                 break_condition: Callable[[np.ndarray], bool] = None,
                 target_count: int = 1,
                 mask_value = None) -> None:
        if len(headers) != data.shape[0]:
            raise ValueError(f'Header list length does not match data array '
                             f'size: {len(headers)} != {data.shape[0]}')
        self.headers = headers
        if len(data.shape) != 2 or data.shape[0] != data.shape[1]:
            raise ValueError(f'Data array needs to be square. Got: {data.shape}')
        if mask_value is not None:
            print(f'Masking value: {mask_value}')
            self.data = ma.masked_equal(data, mask_value)
        else:
            self.data = ma.masked_array(data)
        # No values masked so intialize full mask.
        if isinstance(self.data.mask, ma.MaskType):
            self.data.mask = ma.make_mask_none(self.data.shape)
        self.score = score
        self.summary = summary
        self.break_condition = break_condition
        self.target_count = target_count
        start_summary = None
        if summary:
            start_summary = summary(data)
        self.steps = [(start_summary, '', -1)]

    @staticmethod
    def mask_rowcol(m: ma.MaskedArray, k: int = 0) -> None:
        """Mask the `k`-th row and column of array `m` in place.
        """
        start = time.time_ns()
        m.mask[:,k] = True
        m.mask[k,:] = True
        print(f'mask_entry: {(time.time_ns() - start) / 1000000:.2f} ms',
              end=' ')

    @staticmethod
    def measure(pref: str, start: int) -> int:
        """Print elapsed time in milliseconds since `start`,
        prefixed with `pref`, and return the current time.
        """
        end = time.time_ns()
        print(f'{pref}: {(end - start) / 1000000:.2f} ms', end=' ')
        return end

    def process(self) -> None:
        """Process the input matrix by removing row-column pairs
        in order determined by the specified score function.

        Notes
        -----
        By default the entire matrx is processed, i.e., until only
        one row-column pair is left.
        If a `target_count` is specified, processing aborts when
        `target_count` row-column pairs are left.
        If a `break_condition` is specified, processing aborts when
        the condition is fulfilled.
        """
        data_len = len(self.headers)

        while data_len > self.target_count:
            all_start = time.time_ns()
            rem_idx = self.score(self.data)
            if rem_idx < 0:
                print(f'No valid values left. Aborting early.')
                break
            start = self.measure('score', all_start)

            self.mask_rowcol(self.data, rem_idx)
            start = self.measure('mask_rowcols', start)
            data_len -= 1

            summary_value = None
            if self.summary:
                summary_value = self.summary(self.data)
            start = self.measure('summary', start)
            self.steps.append((summary_value, self.headers[rem_idx], rem_idx))

            if self.break_condition and self.break_condition(self.data):
                break

            end = time.time_ns()
            print(f'total: {(end - all_start) / 1000000:.2f} ms')
            print(self.steps[-1])
