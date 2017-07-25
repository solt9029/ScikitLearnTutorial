import predict_digit as pd
import os, sys, math


if __name__=='__main__':
    if len(sys.argv) <= 2:
        print("Please specify more than two paths of the number images you want to plus in this program as arguments.")
        print("Example: python3 plus.py ./test/test9.png ./test/test4.png")
        exit()

    result=0
    for i in range(len(sys.argv)-1):
        result+=pd.predict_digit(sys.argv[i+1])

    print("RESULT:",result)