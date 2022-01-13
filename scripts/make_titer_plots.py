import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import seaborn as sns


# Plot aesthetics.
plt.style.use('seaborn-deep')
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.alpha'] = 0.2
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'


def sci_notation(n, sig_fig=2):
    if n > 9999:
        fmt_str = '{0:.{1:d}e}'.format(n, sig_fig)
        n, exp = fmt_str.split('e')
        # return n + ' x 10^' + str(int(exp))
        return r'${n:s} \times 10^{{{e:d}}}$'.format(n=n, e=int(exp))
    return str(n)


# Figure 4.
preds = np.array([4.834, 3.793, 2.044, -1.91, -5.84])
titers = np.array([8.70e11, 1.82e12, 1.72e10, 1.48e7, 1.83e4])
fig, ax = plt.subplots(figsize=(4, 3))
ax.scatter(preds, titers, s=10)
ax.set_yscale('log')
ax.set_ylabel(r'Viral Genome (vg/$\mu$L)')
ax.set_xlabel('Predicted Log Enrichment')
ax.set_title('Model Predicted Sequences')
for p, t in zip(preds, titers):
    if p < -5:
        label = '({:0.2f}, {}, Global Min)'
    elif p > 4:
        label = '({:0.2f}, {}, Global Max)'
    else:
        label = '({:0.2f}, {})'
    if p < 2:
        ha = 'left'
        xytext=(10, 0)
    else:
        ha = 'right'
        xytext=(-10, 0)
    plt.annotate(label.format(p, sci_notation(t)), (p, t), textcoords='offset pixels', xytext=xytext, fontsize='xx-small', ha=ha)
ax.grid(False)
plt.tight_layout()
plt.savefig('plots/figure_4.png', dpi=300, transparent=False, bbox_inches='tight', facecolor='white')
plt.close()


# Figure 5A.
labels = ['Library D2', 'Library D3', 'NNK']
preds = np.array([1.909, 0.960, -1.002])
titers = np.array([5.12e11, 2.75e11, 1.02e11])
sds = np.array([23000000000, 18700000000, 9900000000])
fig, ax = plt.subplots(figsize=(3, 3))
ax.scatter(preds, titers, s=10)
ax.errorbar(preds, titers, yerr=sds, fmt='none')
# ax.set_yscale('log')
ax.set_ylabel('Viral Genome (vg/mL)')
ax.set_xlabel('Predicted Log Enrichment')
ax.set_title('Experimental Titer vs. Prediction', pad=20)
trans = mtransforms.ScaledTranslation(-60/72, 20/72, fig.dpi_scale_trans)
ax.text(0.0, 1.0, 'a', transform=ax.transAxes + trans, fontsize='medium', va='bottom', fontfamily='serif')
for i, lbl in enumerate(labels):
    plt.annotate(lbl, (preds[i], titers[i]), textcoords='offset pixels', xytext=(10, 0), fontsize='xx-small', ha='left')
ax.grid(False)
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))   # set position of y spine to y=0
ax.yaxis.set_label_coords(-0.1, 0.5)
plt.tight_layout()
plt.savefig('plots/figure_5a.png', dpi=300, transparent=False, bbox_inches='tight', facecolor='white')
plt.close()


# Figure 5C.
labels = ['Library D2', 'Library D3', 'NNK', 'NNK-Post']
titers = np.array([5.12e11, 2.75e11, 1.02e11, 4.38e11])
sds = np.array([23000000000, 18700000000, 9900000000, 13000000000])
fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(np.arange(len(labels)), titers, yerr=sds, align='center', ecolor='black', capsize=10)
plt.annotate('**', (len(labels)-1, titers[-1]), textcoords='offset pixels', xytext=(0, 40), ha='center', fontsize='small')
ax.grid(False)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, ha='center', fontsize=8)
ax.set_ylabel('Viral Genome (vg/mL)')
ax.set_title('c', fontfamily='serif', loc='left', fontsize='medium', pad=20)
plt.tight_layout()
plt.savefig('plots/figure_5c.png', dpi=300, transparent=False, bbox_inches='tight', facecolor='white')
plt.close()
