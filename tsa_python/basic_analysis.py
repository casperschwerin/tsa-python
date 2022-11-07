import matplotlib.pyplot as plt
import statsmodels.api as sm


def basic_analysis(samples):
    fig, ax = plt.subplots(1, 3)
    ax[0].plot(samples)
    ax[0].axhline(1.96, linestyle="dashed", color="red")
    ax[0].axhline(-1.96, linestyle="dashed", color="red")

    acf = sm.tsa.acf(samples)
    # print(list(range(1, len(acf) + 1)))
    ax[1].set_title("ACF")
    ax[1].scatter(list(range(1,len(acf))), acf[1:])
    ax[1].axhline(2/(len(samples)**0.5), linestyle="dashed", color="red")
    ax[1].axhline(-2/(len(samples)**0.5), linestyle="dashed", color="red")

    pacf = sm.tsa.pacf(samples)
    ax[2].set_title("PACF")
    ax[2].scatter(list(range(1,len(pacf))), pacf[1:])
    ax[2].axhline(2/(len(samples)**0.5), linestyle="dashed", color="red")
    ax[2].axhline(-2/(len(samples)**0.5), linestyle="dashed", color="red")
    plt.show()