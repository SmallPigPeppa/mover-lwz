import matplotlib.pyplot as plt

def test():
    # fig, ax = plt.subplots(
    #     nrows=2,
    #     ncols=5,
    #     gridspec_kw={
    #         # 'width_ratios': [20,]*4 + [1,],
    #         # 'height_ratios': [1,1],
    #         # 'wspace': 0.025,
    #         # 'hspace': 0.025,
    #     },
    #     figsize=(20,4)
    # )
    #
    # gs = ax[0, 0].get_gridspec()
    #
    # # remove the underlying axes
    # for a in ax[:, -1]:
    #     a.remove()
    #
    # ax_0 = fig.add_subplot(gs[:, -1])
    #
    # plt.show()

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 5)
    ax_inp_image = fig.add_subplot(gs[:, 0])

    ax_spin_1 = fig.add_subplot(gs[0, 1])
    ax_spin_2 = fig.add_subplot(gs[0, 2])
    ax_spin_h = fig.add_subplot(gs[0, 3])

    ax_pare_1 = fig.add_subplot(gs[1, 1])
    ax_pare_2 = fig.add_subplot(gs[1, 2])
    ax_pare_h = fig.add_subplot(gs[1, 3])

    ax_colbar = fig.add_subplot(gs[:,-1])

    # f3_ax1 = fig3.add_subplot(gs[0, :])
    # f3_ax1.set_title('gs[0, :]')
    # f3_ax2 = fig3.add_subplot(gs[1, :-1])
    # f3_ax2.set_title('gs[1, :-1]')
    # f3_ax3 = fig3.add_subplot(gs[1:, -1])
    # f3_ax3.set_title('gs[1:, -1]')
    # f3_ax4 = fig3.add_subplot(gs[-1, 0])
    # f3_ax4.set_title('gs[-1, 0]')
    # f3_ax5 = fig3.add_subplot(gs[-1, -2])
    # f3_ax5.set_title('gs[-1, -2]')

    plt.show()


if __name__ == '__main__':
    test()