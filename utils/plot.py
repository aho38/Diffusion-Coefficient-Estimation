
import matplotlib.pyplot as plt
import numpy as np


def plot_step(u1, u2, ud1, ud2, m, m_true = None, gamma = 0, beta1 = 0, beta2 = 0):
    x = np.linspace(0, 1, len(u1))

    fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(17,4),sharey=False)

    ax[0].plot(x, ud1 , 'o-',label=r'$u_{d1}$', markersize=3)
    ax[0].plot(x, u1, '', label=r'$u_{sol}$',linewidth=2.5)
    ax[0].grid('on')
    ax[0].set_xlabel(r'$x$', fontsize=13)
    ax[0].set_ylabel(r'$u$', fontsize=13)
    ax[0].set_title(r'$u_{true1}$ vs. $u_{sol1}: \beta_1$' + f' $ = {beta1:1.1e}$', fontsize=15)
    ax[0].legend(prop={'size':15})

    ax[1].plot(x, ud2 , 'o-',label=r'$u_{d1}$', markersize=3)
    ax[1].plot(x, u2 , '', label=r'$u_{sol}$',linewidth=2.5)
    ax[1].grid('on')
    ax[1].set_xlabel(r'$x$', fontsize=13)
    ax[1].set_ylabel(r'$u$', fontsize=13)
    ax[1].set_title(r'$u_{true2}$ vs. $u_{sol2}: \beta_2$' + f' $ = {beta2:1.1e}$', fontsize=15)
    ax[1].legend(prop={'size':15})

    text = ax[2].yaxis.get_offset_text()
    text.set_fontsize(8)
    ax[2].plot(x, np.exp(m), 'cornflowerblue', label=r'$exp(m_{sol}$)',linewidth=2.5)
    if m_true is not None:
        ax[2].plot(x, np.exp(m_true), '--', c='coral', label=r'$exp(m_{true}$)',linewidth=2.5)
    ax[2].grid('on')
    ax[2].set_xlabel(r'$x$', fontsize=13)
    ax[2].set_ylabel(r'$e^m$', fontsize=13)
    ax[2].set_title(r'$e^{m_{sol}}$ with '+f'$\gamma = {gamma:1.1e}$', fontsize=15)
    ax[2].legend(prop={'size':15})

    # plt.savefig(f'./log/img/dual_helm_synth_{beta1:1.1f}_{beta2:1.1f}.eps', format='eps',dpi=250)
    plt.show()