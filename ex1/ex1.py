import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import scipy
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score,recall_score
import mglearn


def set_data_globaly(path='ex1/data.mat'):
    global dff, centers, stimuli
    # Load Data
    file = scipy.io.loadmat(path)
    data = file['data']
    dff = data['dff'][0][0]
    centers = data['centers'][0][0]
    stimuli = data['stimuli'][0][0]
    return dff, centers, stimuli

def get_stimuli_onsets(start, end, SR=3):
    stim_idxs = np.nonzero((stimuli[:, 1] > start) & (stimuli[:, 1] < end))[0]
    stim_onsets = stimuli[stim_idxs, 1]
    stim_off = np.minimum(stim_onsets + 1.5 * SR, np.ones(len(stim_onsets)) * end)
    return stim_onsets, stim_off, stim_idxs

def q_1_a(neurons=[0, 200, 400, 600, 1080], start=5000, end=8000, ddf=None):
    fig, axs = plt.subplots(5, sharex=True, sharey=True)
    t = range(start, end)
    n = len(neurons)
    stim_onsets, stim_off, stim_idxs = get_stimuli_onsets(start, end)
    
    for i,neuron in enumerate(neurons):
        axs[i].plot(t, ddf[neuron, start:end], label=f'Neuron #{neuron}', color=plt.cm.jet(i/n))
        for j in range(len(stim_idxs)):
            axs[i].axvspan(stim_onsets[j], stim_off[j], alpha=0.2, color='gray')
        axs[i].legend()

    axs[-1].set_xlabel('Frame #')
    axs[0].set_ylabel('Calcium Signal')
    fig.suptitle('Q1A - Neural Signals')
    
    return fig, axs

def q_1_b(start=5000, end=8000, ddf=None):
    fig, ax = plt.subplots()
    capped_data = np.where(dff > 5, 5, dff)
    im = ax.matshow(capped_data[:, start:end], extent=[start, end, 0, capped_data.shape[0]])
    ax.set_xlabel('Frame #')
    ax.set_ylabel('Neuron #')
    cbar = plt.colorbar(im, orientation='vertical')
    cbar.set_label('Calcium Signal')
    fig.suptitle('Q1B - Population activity')

    return fig, ax

def q_1_c(start=5000, end=6000, ddf=None):
    fig, ax = q_1_b(start, end, ddf)
    stim_onsets, stim_off, _ = get_stimuli_onsets(start, end)

    for i, _ in enumerate(stim_onsets):
        ax.axvspan(stim_onsets[i], stim_off[i], alpha=0.2, color='orange')
    
    fig.suptitle('Q1C - Population activity (zoomed in)')
    return fig, ax

def q_1_d(dff, stimuli):
    stim_onsets = stimuli[:, 1]
    avg_stim_res = np.zeros((dff.shape[0], stim_onsets.size))

    for i, onset_f in enumerate(stim_onsets):
        onset = int(onset_f)
        avg_stim_res[:, i] = np.mean(dff[:, onset:onset + 9], axis=1)
    
    return avg_stim_res

def q_2_a(stimuli, avg_stim_res, neurons=[0, 200, 400, 600, 1080]):
    neurons = np.array(neurons) if isinstance(neurons, list) else neurons
    avg_stim_res_example = avg_stim_res[neurons]
    directions = np.unique(stimuli[:, 0])
    example_means = np.zeros((neurons.size, directions.size))
    example_std = np.zeros((neurons.size, directions.size))
    
    for i,direction in enumerate(directions):
        dir_idxs = (stimuli[:, 0]==direction)
        resp_by_dir = avg_stim_res_example[:, dir_idxs]
        example_means[:, i] = np.mean(resp_by_dir, axis=1)
        example_std[:, i] = np.std(resp_by_dir, axis=1)
    
    fig, axs = plt.subplots(neurons.size, sharex=True, sharey=False)
    axs = np.array([axs]) if neurons.size == 1 else axs
    for i in range(neurons.size):
        axs[i].errorbar(
            directions, example_means[i], example_std[i], 
            label=f'Neuron #{neurons[i]}', linestyle='None', 
            marker='o', color=plt.cm.jet(i/neurons.size)
        )
        axs[i].legend()

    axs[-1].set_xlabel('Azimuth (degrees)')
    axs[0].set_ylabel('Mean Calcium Signal')
    axs[-1].set_ylabel('Mean Calcium Signal')
    fig.suptitle('Q2A - Neuronal Tuning Curves')
    
    return fig, axs

def get_prefered_stim_per_n(avg_stim_res):
    directions = np.unique(stimuli[:, 0])
    mean_stim_response = np.zeros((avg_stim_res.shape[0], directions.size))
    
    for i, direction in enumerate(directions):
        curr_stim_ind = np.nonzero(stimuli[:, 0] == direction)[0]
        mean_stim_response[:, i] = np.mean(avg_stim_res[:, curr_stim_ind], axis=1)

    prefered_stim_per_n = np.argmax(mean_stim_response, axis=1)
    return prefered_stim_per_n

def q_2_b(avg_stim_res):
    directions = np.unique(stimuli[:, 0])
    prefered_stim_per_n = get_prefered_stim_per_n(avg_stim_res)
    preffered_stim_sum = np.zeros(directions.shape)

    for i, direction in enumerate(directions):
        preffered_stim_sum[i] = len(np.nonzero(prefered_stim_per_n == i)[0])

    preffered_stim_percents = preffered_stim_sum / preffered_stim_sum.sum()
    fig, ax = plt.subplots()    
    ax.bar(directions, preffered_stim_percents, width=14, align='center', tick_label=directions)

    ax.set_xlabel('Azimuth (degrees)')
    ax.set_ylabel('Proportion of Neurons')
    fig.suptitle('Q2B - Proportion of Neurons Per Stimulus')
    
    return fig, ax

def spatial_plot_helper(ax, x, y, avg_stim_res, stimuli):
    prefered_stim_per_n = get_prefered_stim_per_n(avg_stim_res)
    cmap = plt.get_cmap('plasma')
    norm = Normalize(vmin=prefered_stim_per_n.min(), vmax=prefered_stim_per_n.max())
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    ax.scatter(
        x, y, c=prefered_stim_per_n, 
        cmap=cmap, norm=norm, s=10, alpha=0.9
    )
    
    directions = np.unique(stimuli[:, 0])
    cbar = plt.colorbar(sm, ax=ax, label='Azimuth (degrees)')
    cbar.set_ticks(np.arange(directions.size))
    cbar.set_ticklabels(directions)

    return ax

def q_2_c(avg_stim_res):
    fig, ax = plt.subplots()
    ax = spatial_plot_helper(ax, centers[:, 0], centers[:, 1], avg_stim_res, stimuli)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.suptitle('Q2C - Spatial Map of Neuronal Tuning')
    return fig, ax

def q_3_a(avg_stim_res):
    normalized_avg_stim_res = zscore(avg_stim_res, axis=1)
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(normalized_avg_stim_res)

    fig, ax = plt.subplots()
    ax = spatial_plot_helper(ax, pca_res[:, 0], pca_res[:, 1], avg_stim_res, stimuli)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ttl = f"Q3A - Spatial Map of PCA\n"
    ttl += f"PC1: {(100 * pca.explained_variance_ratio_[0]):.2f}%, "
    ttl += f"PC2: {(100 * pca.explained_variance_ratio_[1]):.2f}%"
    fig.suptitle(ttl)

    print(f'pca_res.shape: {pca_res.shape}')
    
    return fig, ax, pca_res

def q_3_b():
    avg_stim_res = q_1_d(dff, stimuli)
    trasposed_avg_stim_res = avg_stim_res.copy().T
    zscored_trasposed_avg_stim_res = zscore(trasposed_avg_stim_res, axis=1)
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(zscored_trasposed_avg_stim_res)

    fig, ax = plt.subplots()

    cmap = plt.get_cmap('plasma')
    norm = Normalize(vmin=-75, vmax=75)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    ax.scatter(
        pca_res[:, 0], pca_res[:, 1], 
        c=stimuli[:, 0].astype(int), 
        cmap=cmap, norm=norm, s=10
    )

    cbar = plt.colorbar(sm, ax=ax, label='Azimuth (degrees)')
    cbar.set_ticks(np.arange(-75, 76, 15))
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ttl = f"Q3B - Spatial Map of PCA (Transposed Input)\n"
    ttl += f"PC1: {(100 * pca.explained_variance_ratio_[0]):.2f}%, "
    ttl += f"PC2: {(100 * pca.explained_variance_ratio_[1]):.2f}%"
    fig.suptitle(ttl)

    return fig, ax

def q_3_c():
    avg_stim_res = q_1_d(dff, stimuli)
    trasposed_avg_stim_res = avg_stim_res.copy().T
    zscored_trasposed_avg_stim_res = zscore(trasposed_avg_stim_res, axis=1)
    tsne = TSNE(n_components=2)
    tnse_res = tsne.fit_transform(zscored_trasposed_avg_stim_res)
    cmap = plt.get_cmap('plasma')
    norm = Normalize(vmin=-75, vmax=75)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig, ax = plt.subplots()
    ax.scatter(
        tnse_res[:, 0], tnse_res[:, 1], 
        c=stimuli[:, 0].astype(int), cmap=cmap, norm=norm, s=30,
        edgecolors='k', linewidths=0.5, alpha=0.8)
    cbar = plt.colorbar(sm, ax=ax, label='Azimuth (degrees)')
    cbar.set_ticks(np.arange(-75, 76, 15))
    ax.set_xlabel('tSNE axis 1')
    ax.set_ylabel('tSNE axis 2')
    fig.suptitle(f'Q3C - Spatial Map of tSNE 2D Space')

    return fig, ax

def q_4_a_N_b(
    avg_stim_res=None, pca_res=None, prefered_stim_per_n=None, 
    stimuli=None, stim_a=-75, stim_b=75
):
    directions = np.unique(stimuli[:, 0])

    a_idx = np.nonzero(directions == stim_a)[0][0]
    b_idx = np.nonzero(directions == stim_b)[0][0]
    
    if prefered_stim_per_n is None:
        prefered_stim_per_n = get_prefered_stim_per_n(avg_stim_res)
    
    if pca_res is None:
        _, _, pca_res = q_3_a(avg_stim_res)
        plt.close()
    
    idxs_minus_75 = np.nonzero(prefered_stim_per_n == a_idx)[0]
    idxs_75 = np.nonzero(prefered_stim_per_n == b_idx)[0]
    pc1_minus_75 = pca_res[idxs_minus_75, :]
    pc1_75 = pca_res[idxs_75, :]
    targets_minus_75 = np.zeros(idxs_minus_75.size)
    targets_75 = np.ones(idxs_75.size)

    X = np.concatenate((pc1_minus_75, pc1_75))
    y = np.concatenate((targets_minus_75, targets_75))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    fig4a, ax4a = plt.subplots()

    mglearn.plots.plot_2d_separator(lda, X, fill=True, eps=0.5, alpha=0.4, ax=ax4a)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax4a)
    ax4a.set_xlabel('PC1')
    ax4a.set_ylabel('PC2')
    fig4a.suptitle(
        f'Q4A - LDA Decision Boundary For Stimuli {stim_a} and {stim_b}'
    )

    # Q4b

    y_pred = lda.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=[f'{stim_a}', f'{stim_b}']
    )
    
    cm_disp.plot()
    cm_disp.figure_.suptitle(
        f'Q4B - Confusion Matrix (LDA) For Stimuli {stim_a} and {stim_b}'
    )
    # Q4c

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return fig4a, ax4a, cm_disp, cm, lda, accuracy, recall
    
    

if __name__ == '__main__':
    dff, centers, stimuli = set_data_globaly()
    
    # Q1    
    fig1a, axs1a = q_1_a(ddf=dff)
    plt.show()
    
    fig1b, axs1b = q_1_b(ddf=dff)
    plt.show()

    fig1c, axs1c = q_1_c(ddf=dff)
    plt.show()

    avg_stim_res = q_1_d(dff, stimuli)

    # Q2
    fig2a, axs2a = q_2_a(stimuli, avg_stim_res)
    plt.show()
    
    fig2b, ax2b = q_2_b(avg_stim_res)
    plt.show()

    fig2c, ax2c = q_2_c(avg_stim_res)
    plt.show()

    # Q3

    fig3a, ax3, pca_res = q_3_a(avg_stim_res)
    plt.show()  

    fig3b, ax3b = q_3_b()
    plt.show()
    
    fig3c, ax3c = q_3_c()
    plt.show()

    # Q4
    # Q4 A+B+C
    fig4a, ax4a, cm_disp, cm, lda, accuracy, recall  = q_4_a_N_b(
        avg_stim_res=avg_stim_res, pca_res=pca_res,
        stimuli=stimuli, stim_a=-75, stim_b=75
    )
    print(f'Q4C: Accuracy: {accuracy:.2f}, Recall: {recall:.2f}')

    # Q4 D
    fig4d, ax4d, cm_disp_d, cm_d, lda_d, accuracy_d, recall_d = q_4_a_N_b(
        avg_stim_res=avg_stim_res, pca_res=pca_res,
        stimuli=stimuli, stim_a=-75, stim_b=-60
    )
    fig4d.suptitle('Q4D - LDA Decision Boundary For Stimuli -75 and -60')
    cm_disp_d.figure_.suptitle('Q4D - Confusion Matrix (LDA) For Stimuli -75 and -60')
    print(f'Q4D: Accuracy: {accuracy_d:.2f}, Recall: {recall_d:.2f}')
    plt.show()
    
    

    



