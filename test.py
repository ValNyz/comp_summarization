#!/usr/python3 

def test(*param):
    print(len(param))
    for p in param:
        print(p)

if __name__=='__main__':
    a = (1, 2, 3, 4)
    b = (5, 6)
    test(a)
    test(a, b)
    test(b)
