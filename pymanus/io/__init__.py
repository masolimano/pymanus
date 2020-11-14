import os

def next_path(path_pattern):
    """
    Finds the next free path in an sequentially named list of files

    e.g. path_pattern = 'file-%s.txt':

    file-1.txt
    file-2.txt
    file-3.txt

    Runs in log(n) time where n is the number of existing files in sequence

    Note: Snippet stolen from `James` on SO/17984809
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

if __name__ == '__main__':
    print(next_path('peo{:02d}.txt'))
