import numpy as np

class Environment:
    def __init__(self, M=2, N=3, F=5, K=2, 
                 C_cache=3, C_rec=2,
                 alpha=0.5, D_max=0.2, phi=100,
                 trans_power_factor=1.0, 
                 BH_cost=1.0, 
                 seed=None):
        """
        Environment for joint content caching, recommendation, and user association.
        State includes video preference distribution (VPD) and wireless channel state (WCSI).
        Action includes caching decisions (X), recommendation decisions (Y), and user association (Z).
        Reward is negative total energy consumption, with penalty for exceeding preference deviation.
        Parameters:
            M: number of base stations
            N: number of users
            F: number of video contents
            K: number of quality layers per video (including base layer)
            C_cache: cache capacity (number of (video,layer) items that can be cached per BS)
            C_rec: recommendation list size per BS (number of (video,layer) items that can be recommended per BS)
            alpha: recommendation influence factor (fraction by which recommendations shape user preferences)
            D_max: tolerance threshold for average preference deviation
            phi: penalty factor when preference deviation exceeds D_max
            trans_power_factor: scaling factor for transmission energy (to compute E_trans)
            BH_cost: backhaul energy cost per content unit (for delivering a layer not cached)
            seed: random seed for reproducibility
        """
        self.M = M
        self.N = N
        self.F = F
        self.K = K
        self.C_cache = C_cache
        self.C_rec = C_rec
        self.alpha = alpha
        self.D_max = D_max
        self.phi = phi
        self.trans_power_factor = trans_power_factor
        self.BH_cost = BH_cost
        # Initialize user video preference distribution p[n,f,k] and channel state gamma[m,n]
        rng = np.random.RandomState(seed) if seed is not None else np.random
        # Initial preferences: random distribution for each user (sum of probabilities =1)
        self.p = np.zeros((N, F, K))
        base_pref = rng.rand(N, F)
        base_pref = base_pref / base_pref.sum(axis=1, keepdims=True)
        # Assign base layer preferences and diminishing weights for higher layers
        self.p[:, :, 0] = base_pref
        for k in range(2, K+1):
            self.p[:, :, k-1] = base_pref / (2**(k-2) * K)
        self.p = self.p / self.p.sum(axis=(1,2), keepdims=True)  # normalize distribution per user
        # Wireless channel states (random initial values in (0.2, 1.0] for each BS-user link)
        self.gamma = rng.rand(M, N) * 0.8 + 0.2
        # Cache state (which content layers are currently cached at each BS), initially empty
        self.cache_state = np.zeros((M, F, K), dtype=int)
        self.t = 0  # time step counter

    def reset(self):
        """Reset environment for a new episode. Returns initial state."""
        rng = np.random
        base_pref = rng.rand(self.N, self.F)
        base_pref = base_pref / base_pref.sum(axis=1, keepdims=True)
        # Reinitialize preference distribution
        self.p[:, :, 0] = base_pref
        for k in range(2, self.K+1):
            self.p[:, :, k-1] = base_pref / (2**(k-2) * self.K)
        self.p = self.p / self.p.sum(axis=(1,2), keepdims=True)
        # Randomize channel state
        self.gamma = rng.rand(self.M, self.N) * 0.8 + 0.2
        # Reset cache (empty)
        self.cache_state.fill(0)
        self.t = 0
        return self.get_state()

    def get_state(self):
        """Return current state as a flat array [p, gamma]."""
        state = np.concatenate([self.p.flatten(), self.gamma.flatten()])
        return state.astype(np.float32)

    def step(self, X, Y, Z):
        """
        Apply action (X: caching, Y: recommendation, Z: association).
        X, Y, Z are binary arrays of shapes (M,F,K), (M,F,K), (M,N) respectively.
        Returns (next_state, reward, done, info).
        """
        # Compute post-recommendation preference distribution tilde_p for each user
        tilde_p = np.copy(self.p)
        for n in range(self.N):
            m = np.argmax(Z[:, n])  # BS associated with user n
            R_indices = np.argwhere(Y[m] == 1)  # recommended content indices at BS m
            if R_indices.size > 0:
                L = R_indices.shape[0]
                # Adjust preferences: recommended items get a portion alpha/L, others scaled by (1-alpha)
                tilde_sum = 0
                for f in range(self.F):
                    for k in range(self.K):
                        if Y[m, f, k] == 1:
                            tilde_p[n, f, k] = self.alpha * (1.0 / L) + (1 - self.alpha) * self.p[n, f, k]
                        else:
                            tilde_p[n, f, k] = (1 - self.alpha) * self.p[n, f, k]
                        tilde_sum += tilde_p[n, f, k]
                if abs(tilde_sum - 1.0) > 1e-6:  # normalize to sum=1 (numerical check)
                    tilde_p[n] /= tilde_sum
            else:
                tilde_p[n] = np.copy(self.p[n])  # no change if no recommendations

        # Energy consumption calculation:
        # Caching energy (if new content is fetched into cache this step)
        new_cache = (X == 1) & (self.cache_state == 0)
        E_cache = self.BH_cost * np.sum(new_cache)  # cost per newly cached item
        self.cache_state = X.copy()  # update cache state

        # Content delivery energy
        E_delivery = 0.0
        total_delivered_layers = 0.0
        total_cached_layers = 0.0
        for n in range(self.N):
            m = np.argmax(Z[:, n])
            # Iterate over each content (f) and quality request (k layers) that user n might request
            for f in range(self.F):
                for k in range(1, self.K+1):
                    prob_request = tilde_p[n, f, k-1]  # probability user n requests video f with quality k
                    if prob_request == 0:
                        continue
                    # If user requests k layers, deliver layers 1..k:
                    for i in range(1, k+1):
                        cached = (X[m, f, i-1] == 1)
                        gamma_mn = self.gamma[m, n] if self.gamma[m, n] > 0 else 1e-3
                        E_trans = self.trans_power_factor / gamma_mn  # transmission energy for this layer
                        if cached:
                            E_delivery += prob_request * E_trans
                            total_cached_layers += prob_request
                        else:
                            E_delivery += prob_request * (E_trans + self.BH_cost)
                        total_delivered_layers += prob_request

        E_total = E_cache + E_delivery  # total energy this step

        # Preference deviation D^t
        diff = tilde_p - self.p
        D = np.mean(np.square(diff).sum(axis=(1,2)))  # average quadratic deviation across users

        # Reward (negative energy, with penalty if deviation > D_max)
        if D <= self.D_max:
            reward = -E_total
        else:
            reward = -E_total - self.phi * (D - self.D_max)

        # Prepare info dict with additional metrics
        info = {
            "E_total": E_total,
            "E_cache": E_cache,
            "E_delivery": E_delivery,
            "D": D,
            "cache_hit_rate": (total_cached_layers / total_delivered_layers) if total_delivered_layers > 0 else 0.0
        }

        # Update environment for next state:
        # Blend preferences slightly towards tilde_p to simulate preference change
        blend = 0.1
        self.p = (1 - blend) * self.p + blend * tilde_p
        self.p = self.p / self.p.sum(axis=(1,2), keepdims=True)
        # Evolve channel states (random drift)
        self.gamma = 0.9 * self.gamma + 0.1 * np.random.rand(self.M, self.N)
        self.t += 1

        done = False  # (We use fixed episode length, so done handled outside)
        next_state = self.get_state()
        return next_state, reward, done, info
