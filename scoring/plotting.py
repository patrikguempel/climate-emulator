import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

colors = ["#0072B2", "#E69F00", "#2B2B2B", "#009E73", "#D55E00", "#882255"]

def createPlots(models):
    fn_metrics = {model: f"models/{model}/evaluation/metrics.lev-avg.csv" for model in models}
    fn_metrics_stacked = {model: f"models/{model}/evaluation/metrics.csv" for model in models}
    lc_model = {model: colors.pop() for model in models}

    var_short_name = {'ptend_t': 'dT/dt',
                      'ptend_q0001': 'dq/dt',
                      'cam_out_NETSW': 'NETSW',
                      'cam_out_FLWDS': 'FLWDS',
                      'cam_out_PRECSC': 'PRECSC',
                      'cam_out_PRECC': 'PRECC',
                      'cam_out_SOLS': 'SOLS',
                      'cam_out_SOLL': 'SOLL',
                      'cam_out_SOLSD': 'SOLSD',
                      'cam_out_SOLLD': 'SOLLD',
                      }

    var_idx = {}
    var_idx['ptend_t'] = (0, 60)
    var_idx['ptend_q0001'] = (60, 120)
    var_idx['surface_vars'] = (120, 128)

    plot_this_models = models
    plot_this_models_crps = []
    plot_this_metrics = ['MAE', 'RMSE', 'R2']
    abc = 'abcdefg'

    PLOTDATA = {}
    for kmodel in plot_this_models:
        # MSE, R2, RMSE
        PLOTDATA[kmodel] = pd.read_csv(fn_metrics[kmodel], index_col=0)

    PLOTDATA_by_METRIC = {}
    for kmetric in plot_this_metrics:
        if kmetric in ['CRPS']:
            k_plot_this_models = plot_this_models_crps
        if kmetric in ['MAE', 'R2', 'RMSE']:
            k_plot_this_models = plot_this_models
        PLOTDATA_by_METRIC[kmetric] = pd.DataFrame([PLOTDATA[kmodel][kmetric] for kmodel in k_plot_this_models],
                                                   index=k_plot_this_models
                                                   )

    fig, _ax = plt.subplots(nrows=len(plot_this_metrics),
                            sharex=True)

    for k, kmetric in enumerate(plot_this_metrics):
        ax = _ax[k]
        plotdata = PLOTDATA_by_METRIC[kmetric]
        plotdata = plotdata.rename(columns=var_short_name)
        plotdata = plotdata.transpose()
        plotdata.plot.bar(color=[lc_model[kmodel] for kmodel in plotdata.keys()],
                          width=.4 if kmetric == 'CRPS' else .6,
                          legend=False,
                          ax=ax)

        ax.set_title(f'({abc[k]}) {kmetric}')
        ax.set_xlabel('Output variable')
        ax.set_xticklabels(plotdata.index, rotation=0, ha='center')

        # no units for R2
        if kmetric != 'R2':
            ax.set_ylabel('W/m2')

        # not plotting negative R2 values
        if kmetric == 'R2':
            ax.set_ylim(0, 1)

        # log y scale
        if kmetric != 'R2':
            ax.set_yscale('log')

        fig.set_size_inches(7, 8)

    _ax[0].legend(ncols=3, columnspacing=.9, labelspacing=.3,
                  handleheight=.07, handlelength=1.5, handletextpad=.2,
                  borderpad=.2,
                  loc='upper right')

    fig.tight_layout()
    fig.savefig('plots/performace_plot.pdf')

    PLOTDATA = {}
    for kmodel in plot_this_models:
        # MSE, R2, RMSE
        PLOTDATA[kmodel] = pd.read_csv(fn_metrics_stacked[kmodel], index_col=0)

    PLOTDATA_by_METRIC = {}
    for kmetric in plot_this_metrics:
        if kmetric in ['CRPS']:
            k_plot_this_models = plot_this_models_crps
        if kmetric in ['MAE', 'R2', 'RMSE']:
            k_plot_this_models = plot_this_models
        PLOTDATA_by_METRIC[kmetric] = pd.DataFrame([PLOTDATA[kmodel][kmetric] for kmodel in k_plot_this_models],
                                                   index=k_plot_this_models
                                                   )

    abc = 'abcdefg'
    for kvar in ['ptend_t', 'ptend_q0001']:
        fig, _ax = plt.subplots(ncols=2, nrows=2)
        _ax = _ax.flatten()
        for k, kmetric in enumerate(plot_this_metrics):
            ax = _ax[k]
            idx_start = var_idx[kvar][0]
            idx_end = var_idx[kvar][1]
            plotdata = PLOTDATA_by_METRIC[kmetric].iloc[:, idx_start:idx_end]
            if kvar == 'ptend_q0001':
                plotdata.columns = plotdata.columns - 60
            if kvar == 'ptend_q0001':  # this is to keep the right x axis range.
                plotdata = plotdata.where(~np.isinf(plotdata), -999)
            plotdata = plotdata.transpose()
            plotdata.plot(color=[lc_model[kmodel] for kmodel in plotdata.keys()],
                          legend=False,
                          ax=ax,
                          )

            ax.set_xlabel('Level index')
            ax.set_title(f'({abc[k]}) {kmetric} ({var_short_name[kvar]})')
            if kmetric != 'R2':
                ax.set_ylabel('W/m2')

            # R2 ylim
            if (kmetric == 'R2'):
                ax.set_ylim(0, 1.05)

        # legend
        _ax[0].legend(ncols=1, labelspacing=.3,
                      handleheight=.07, handlelength=1.5, handletextpad=.2,
                      borderpad=.3,
                      loc='upper left')

        fig.tight_layout()
        fig.set_size_inches(7, 4.5)
        if kvar == 'ptend_t':
            fig.savefig('plots/ptend_t.pdf')
        elif kvar == 'ptend_q0001':
            fig.savefig('plots/ptend_q0001.pdf')

createPlots(["mlp1", "cnn2", "mb_mlp3", "xgb/d12"])