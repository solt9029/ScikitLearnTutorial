import predict_digit as pd
import os, sys, math

if len(sys.argv) <= 2:
    print("Please specify two paths of the number images you want to plus in this program as second and third arguments.")
    exit()

result=pd.predict_digit(sys.argv[1])+pd.predict_digit(sys.argv[2])

print("RESULT:",result)