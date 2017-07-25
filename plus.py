import predict_digit as pd

import os, sys, math
if len(sys.argv) <= 1:
    print("Please specify the path of the number image you want to predict in this program as second argument.")
    exit()

imagefile=sys.argv[1]
digit=pd.predict_digit(imagefile)
print("RESULT",digit)