from stable_baselines3.common.results_plotter import plot_results
import matplotlib.pyplot as plt
if __name__ == "__main__":
    log_dir = 'result'
    plot_results([log_dir], 100000000, 'episodes', "TD3 LunarLander")
    plt.show()