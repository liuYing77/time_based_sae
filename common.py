import numpy as np


v_t   = np.float32
spk_t = np.int32
idx_t = np.int32

def default_spike_eval(ref_time, run_time, curr_time, spikes):
    max_val = 1.
    dt = float( max(1, curr_time - ref_time) )
    return spikes*(max_val/dt)

def default_w_init(n_rows, n_cols, def_dtype=v_t):
#     max_w = 1./np.sqrt(n_rows*n_cols)
#     return np.random.uniform(-max_w, max_w, size=(n_rows, n_cols)).astype(v_t)
    return np.random.uniform(0.01, 0.5, size=(n_rows, n_cols)).astype(v_t)

def default_stdp(ref_time, run_time, curr_time, spike_to_value,
                 t_minus, t_plus, weights, min_w, max_w, learn_rate,
                 last_pre_spikes, pre_spikes, 
                 post_spikes, target_spikes):
    # post_input = [w]*pre_spikes
    w = weights

    # time weight <==> sooner spikes (t->ref) should be more important    
    tw = spike_to_value(ref_time, run_time, curr_time, 1.)
    ww = tw*learn_rate
    rows = np.where( target_spikes > 0 )[0]
    twin_start = max(0, (curr_time - t_minus))
    
    if len(rows):
        cols = np.where( last_pre_spikes > twin_start)[0]
        if len(cols):
            w[:, cols] += ww*np.outer( target_spikes[:, 0], (curr_time - last_pre_spikes[cols,0]) )

        else:
            w[rows, :] += ww*np.outer( target_spikes[rows, 0], np.ones_like(pre_spikes, dtype=spk_t) )


    elif (post_spikes>0).any():
        cols = np.where( last_pre_spikes > twin_start)[0]
        if len(cols):
            w[cols, :] -= ww*np.outer( post_spikes[:, 0], (curr_time - last_pre_spikes[cols,0]) )


    w[:] = np.clip(w, min_w, max_w)
        
    return w

default_markers = {'in': '^', 'hid': 'o', 'rcn': '+', 'tgt': 'x'}
default_colors  = {'in': 'g', 'hid': 'c', 'rcn': 'm', 'tgt': 'b'}
default_labels  = {'in': 'input', 'hid': 'hidden', 'rcn': 'reconstruct', 'tgt': 'target'}
def plot_spikes(aelevel, run_time, markers=default_markers, colors=default_colors,
                labels=default_labels, figsize=(12, 10), size=8):
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=figsize)
    ax  = plt.subplot(1, 1, 1)
    k = ''
    max_y = max(aelevel._sizes['hid'], aelevel._sizes['in'])
    plt.ylim((-1, max_y+1))
    plt.xlim((-1, int(run_time)+1))
    
    x, y = None, None
    for k in aelevel._spikes:
        y, x = np.where(aelevel._spikes[k] > 0)
        plt.plot(x, y, marker=markers[k], mec=colors[k], linestyle='none',\
                 fillstyle='none', label=labels[k], mew=1, markersize=size)
    
    if hasattr(aelevel, '_target_spikes'):
        k = 'tgt'
        y, x = np.where(aelevel._target_spikes > 0)
        plt.plot(x, y, marker=markers[k], mec=colors[k], linestyle='none',\
                 fillstyle='none', label=labels[k], mew=1, markersize=size)
    plt.legend()