
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

def plot_step_single(u1, ud1, m, m_true = None, gamma = 0):
    x = np.linspace(0, 1, len(u1))

    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(17,4),sharey=False)

    ax[0].plot(x, ud1 , 'o-',label=r'$u_{d1}$', markersize=3)
    ax[0].plot(x, u1, '', label=r'$u_{sol}$',linewidth=2.5)
    ax[0].grid('on')
    ax[0].set_xlabel(r'$x$', fontsize=13)
    ax[0].set_ylabel(r'$u$', fontsize=13)
    ax[0].set_title(r'$u_{true1}$ vs. $u_{sol1}$', fontsize=15)
    ax[0].legend(prop={'size':15})

    ax[1].plot(x, np.exp(m), 'cornflowerblue', label=r'$exp(m_{sol}$)',linewidth=2.5)
    if m_true is not None:
        ax[1].plot(x, np.exp(m_true), '--', c='coral', label=r'$exp(m_{true}$)',linewidth=2.5)
    ax[1].grid('on')
    ax[1].set_xlabel(r'$x$', fontsize=13)
    ax[1].set_ylabel(r'$e^m$', fontsize=13)
    ax[1].set_title(r'$e^{m_{sol}}$ with '+f'$\gamma = {gamma:1.1e}$', fontsize=15)
    ax[1].legend(prop={'size':15})

    # plt.savefig(f'./log/img/dual_helm_synth_{beta1:1.1f}_{beta2:1.1f}.eps', format='eps',dpi=250)
    plt.show()

def plot_mean_shaded(rbest, nfe, color, label=None, title=None, ylabel="R best", xlabel="Number of function evaluations", xlim=None, shade=True, **kwargs):
    if xlim is not None:
        plt.xlim(xlim)
    
    mean = rbest.mean(axis=1)
    std = rbest.std(axis=1)
    plt.plot(nfe, rbest.mean(axis=1), color=color, label=label, **kwargs)
    plt.fill_between(nfe, mean - std, mean + std, alpha=0.25, color=color) if shade else None
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel(xlabel, fontsize = 12)
    plt.ylabel(ylabel, fontsize = 12)
    plt.title(title)