from sklearn import datasets, svm 
from sklearn.externals import joblib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os, sys, math



# 画像から数字を予測し、その数字を返す関数
def predict_digit(imagefile):
    DIGITS_PKL = "./model/digit-clf.pkl"

    # 画像を8×8に変換する処理
    image = Image.open(imagefile).convert('L')
    image = image.resize((8, 8), Image.ANTIALIAS)
    img = np.asarray(image, dtype=float)
    img = np.floor(16 - 16 * (img / 256))
    # 変換された画像を表示する場合はコメントアウトを解除する
    plt.imshow(img)
    plt.gray()
    plt.show()
    img = img.flatten()

    # モデルが作成されてない場合は作る
    if not os.path.exists(DIGITS_PKL):
        digits = datasets.load_digits()

        # 訓練
        data_train = digits.data
        label_train = digits.target
        clf = svm.SVC(gamma=0.001)
        clf.fit(data_train, label_train)

        # 予測モデルを保存
        joblib.dump(clf, DIGITS_PKL)

    clf = joblib.load(DIGITS_PKL)
    n = clf.predict([img])
    return n[0]



if __name__=='__main__':
    if len(sys.argv) <= 1:
        print("Please specify the path of the number image you want to predict in this program as second argument.")
        print("Example: python3 predict_digit.py ./test/test9.png")
        exit()

    imagefile=sys.argv[1]
    result=predict_digit(imagefile)
    print("RESULT:",result)