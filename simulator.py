import numpy as np

v_t   = np.float32
spk_t = np.float32
idx_t = np.int32

class LIF(object):
    def __init__(self, description):
        self._params = description['params']
        self._size   = description['size']
        self._v      = np.zeros((self._size, 1), dtype=v_t)
        self._spikes = np.zeros((self._size, 1), dtype=spk_t)
        self._dv     = np.exp(-1./self._params['tau_m'])
        
        self.reset()
        
    def reset(self):
        self._v[:] = self._params['v_rest']
        self._spikes[:] = 0

    def sim(self, t, in_val):
        if in_val is not None:
            self._v += in_val

        self._v *= self._dv
        
        self._spikes[:] = 0
        spiked = np.where(self._v > self._params['v_thresh'])[0]
        self._spikes[spiked, 0] = 1

        self._v[spiked, 0] = self._params['v_rest']
        
        return self._spikes

class SpikeIn(object):
    def __init__(self, description):
        self._size = description['size']
        self._spikes = np.zeros((self._size, 1), dtype=spk_t)
        self._times = self._parse_description(description)
    
    def _parse_description(self, d):
        times = {}
        for i in range(self._size):
            for t in d['spike_times'][i]:
                if t not in times:
                    times[t] = [i]
                else:
                    times[t].append(i)
        
        return times
    def reinit(self, spike_times):
        self._times.clear()
        self._times = self._parse_description({'spike_times': spike_times})

    def reset(self):
        return 

    def sim(self, t, in_val):
        self._spikes[:] = 0
        if t in self._times:
            self._spikes[self._times[t], 0] = 1

        return self._spikes

def default_spike_eval(ref_time, run_time, curr_time, spikes):
    max_val = 1.
    return spikes*(max_val/(curr_time - ref_time + 1.))

def default_w_init(n_rows, n_cols, def_dtype=v_t):
    max_w = 1./np.sqrt(n_rows*n_cols)
    return np.random.uniform(-max_w, max_w, size=(n_rows, n_cols)).astype(v_t)

def default_stdp(ref_time, run_time, curr_time, spike_to_value,
                 t_minus, t_plus, weights, min_w, max_w, learn_rate,
                 last_pre_spikes, pre_spikes, 
                 post_spikes, target_spikes):
    w = weights
    # time weight <==> sooner spikes (t->ref) should be more important    
    tw = spike_to_value(ref_time, run_time, curr_time, 1.)

#     # weight updates
#     left_bound = max(t-tau_stdp+1-delay, 0)
    if (post_spikes>0).any() or (target_spikes>0).any():

        rows = np.where( np.logical_and(last_pre_spikes > (curr_time - t_minus), last_pre_spikes >= 0) )[0]
        w[rows, :] += learn_rate*np.outer(last_pre_spikes[rows,0], target_spikes[:, 0])*tw
        w[rows, :] -= learn_rate*np.outer(last_pre_spikes[rows,0], post_spikes[:, 0])*tw
        
        cols = np.where( target_spikes > 0 )[0]
        w[:, cols] += learn_rate*np.outer(np.ones_like(pre_spikes, dtype=spk_t), target_spikes[cols, 0])*tw
        cols = np.where( post_spikes > 0 )[0]
        w[:, cols] -= learn_rate*np.outer(np.ones_like(pre_spikes, dtype=spk_t), post_spikes[cols, 0])*tw

        w[w>w_max] = w_max
        w[w<w_min] = w_min

    return w

class AELayer(object):
    def __init__(self, description):
        """ keys for internal layers are 'in', 'hid', 'rcn': 
            input, hidden and reconstruction populations respectively
        """
        
        if description['sizes']['in'] != description['sizes']['rcn']:
            raise Exception("In Autoencoder Layer: Input and Reconstruction sizes differ %d != %d"%\
                            (description['sizes']['in'], description['sizes']['rcn']))
        
        if description['level'] == 0:
            self._in = SpikeIn({'size': description['sizes']['in'], \
                                'spike_times': description['in_times']})
            
        else:
            self._in = LIF({'size': description['sizes']['in'], \
                            'params': description['neuron_params']})
        
        self._hid = LIF({'size': description['sizes']['hid'], \
                         'params': description['neuron_params']})
        
        self._rcn = LIF({'size': description['sizes']['rcn'], \
                         'params': description['neuron_params']})
        
        self._sizes = description['sizes']
        
        self._run_time = description['run_time']
        
        self._delays = {k: np.array(description['delays'][k], dtype=idx_t) \
                                             for k in description['delays']}

        self._spikes = {}
        for k in self._sizes:
            max_len = max(description['run_time'], self._delays[k].max())
            self._spikes[k] = np.zeros((self._sizes[k], max_len))
        
        self._last_spike_times = {k: -1000.*np.ones((self._sizes[k], 1), dtype=idx_t)\
                                                                 for k in self._sizes}
            
        self._v_inputs = {k: np.zeros((self._sizes[k], 1)) for k in self._sizes}
        
        if 'spike_eval' in description:
            self._spike_to_val = description['spike_eval']
        else:
            self._spike_to_val = default_spike_eval

        if 'weight_initialize' in description:
            self._w = description['weight_initialize'](self._sizes['in'],
                                                       self._sizes['hid'], v_t)
        else:
            self._w = default_w_init(self._sizes['in'], self._sizes['hid'],  v_t)

        if 'stdp' in description:
            self._stdp    = description['stdp']['func']
            self._max_w   = description['stdp']['max_w']
            self._min_w   = description['stdp']['min_w']
            self._t_plus  = description['stdp']['t_plus']
            self._t_minus = description['stdp']['t_minus']
            self._rate    = description['stdp']['learn_rate']
            
            max_time = description['run_time'] + description['stdp']['target_delay'] + \
                       np.max(description['delays']['hid']) + 1
            max_time = int(max_time*1.1)
            self._target_spikes = np.zeros((description['sizes']['rcn'], max_time), 
                                           dtype=spk_t)
            for i in range(description['sizes']['rcn']):
                if len(description['stdp']['target_times'][i]):
                    self._target_spikes[i, description['stdp']['target_times'][i]] = 1.
        else:
            self._stdp = None
    
    def retarget(self, target_times):
        self._target_spikes[:] = 0
        for i in range(self._sizes['rcn']):
            if target_times[i]:
                self._target_spikes[i, target_times[i]] = 1.

    def reinput(self, input_times):
        if isinstance(self._in, SpikeIn):
            self._in.reinit(input_times)
        
    def reset(self):
        self._in.reset()
        self._hid.reset()
        self._rcn.reset()

    def store_spikes(self, pop, t, spikes):
        """ Axonal delays. Single delay per output neuron.
            pop: key/index of population
            t: current sim time
            spikes: generated by pop at time t
        """
        rt = self._run_time
        
        idx = t + self._delays[pop]
        idx[idx >= rt] = idx[idx >= rt] - rt

        self._spikes[pop][:, idx] = spikes

    def recover_spikes(self, pop, t):
        return (self._spikes[pop][range(self._sizes[pop]), t - self._delays[pop]]).reshape(-1, 1)
        
    def sim(self, t, in_spikes=None):
        s2v = self._spike_to_val
        rt = self._run_time
        tref = 0.

        self._v_inputs['in'][:] = in_spikes
        self.store_spikes( 'in', t, self._in.sim(t, self._v_inputs['in']) )
        
        self._v_inputs['hid'][:] = np.dot( np.transpose(self._w), self.recover_spikes('in', t) )
        self.store_spikes( 'hid', t, self._hid.sim(t, self._v_inputs['hid']) )
        #s2v( tref, t, rt, self.recover_spikes('in', t) )

        self._v_inputs['rcn'][:] = np.dot( self._w, self.recover_spikes('hid', t) )
        self.store_spikes( 'rcn', t, self._rcn.sim(t, self._v_inputs['rcn']) )
        for k in self._last_spike_times:
            self._last_spike_times[k][ self.recover_spikes(k, t) > 0 ] = t

        if self._stdp is not None:
            self._w.T[:] = self._stdp(tref, rt, t, s2v, self._t_minus, self._t_plus,
                                    self._w.T, self._min_w, self._max_w, self._rate,
                                    self._last_spike_times['hid'],
                                    self.recover_spikes('hid', t), #pre/hidden
                                    self.recover_spikes('rcn', t), #post/reconstruction
                                    self._target_spikes[:, t].reshape(-1, 1))     #target for current time


        return self.recover_spikes('rcn', t)
            

    