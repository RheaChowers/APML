import numpy as np

l = np.array(
    [
        [0,11,12,1],
        [11, 0, 9, 7],
        [12, 9, 0, 5],
        [1, 7, 5, 0]
    ]
)
m = 2

print(l, "\n---------------")

print(l.sum(axis=0))

t = np.array([[1,4,3,2], [1,0,4,6]])

print("blabla\n", np.argsort(t))