import linReg
import numpy as np

def main():
    x = np.array([1,2,3])
    y = np.array([340, 660, 980])
    lr = 0.1
    step = 300
    model = linReg.LinearRegModel()
    model.train(x, y, lr, step)
    for i, val in enumerate(x):
        print(f"Prediction for {val} is {model.predict(val):.2f}, real value is {y[i]}")

if __name__ == "__main__":
    main()