import numpy as np

class A:
    def __init__(self):
        self.train = 3
        self.phi = self.get_phi(self.train)

    def get_phi(self, train):
        b = 5-train
        print("i was called")
        def p(a):
            print("this is a ", a)
            l = []
            if b > a:
                l.append(4)
            else:
                l.append(6)
            return l
        return p

if __name__ == '__main__':
    print([3*i+5 for i in range(3)])
