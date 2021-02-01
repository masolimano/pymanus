import os

def next_path(path_pattern):
    """
    Finds the next free path in an sequentially named list of files.
    Runs in log(n) time where n is the number of existing files in sequence

    Parameters
    ----------
    path_pattern: str
        String containing the pattern of the filename to increment. It has
        to contain a format specifier in the new-style format minilanguage

    Returns
    ---------
    next_path: str
        Next filename in the current folder following the input pattern.

    Note: Snippet stolen from `James` on SO/17984809

    Example usage:
    ---------
    >>> import os
    >>> os.chdir('/tmp/')
    >>> os.mkdir('test-001')
    >>> os.mkdir('test-002')
    >>> next_path('test-{:03d}')
    'test-003'

    """
    i = 1

    # First do an exponential search
    while os.path.exists(path_pattern.format(i)):
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2 # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern.format(c)) else (a, c)

    return path_pattern.format(b)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
