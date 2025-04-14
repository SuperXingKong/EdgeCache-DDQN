import numpy as np

class AdvancedEnvironment:
    """
    A more detailed environment to closely match the paper's System Model:
      - Uses FSMC for user preference distribution (VPD)
      - Uses FSMC for wireless channel states (WCSI)
      - Computes transmission rates using log(1+SINR) model
      - Distinguishes local caching vs. backhaul fetching in energy consumption
      - Handles multi-layer SVC transmission
      - Imposes a penalty if preference deviation D^t exceeds D_max
    """

    def __init__(self,
                 M=2,          # 基站数
                 N=4,          # 用户数
                 F=5,          # 视频总数
                 K=3,          # 每个视频的层数 (1个BL + K-1个EL)
                 C_cache=4,    # 每个基站可缓存的视频层数上限
                 C_rec=2,      # 每个基站向用户推荐的视频层数上限
                 D_max=0.2,
                 phi=100,
                 B=10.0,       # 无线带宽 (MHz 或者任意单位)
                 BH_B=5.0,     # 回程带宽
                 P_m=1.0,      # 基站发送功率 (简化为相同)
                 w_BH=0.5,     # 回程能耗因子 (J/bit) 或 (W/bit)，视实现而定
                 # FSMC 相关参数：用户偏好状态
                 skew_states=[1.0, 1.2, 1.5],  # Zipf偏斜因子可能的取值
                 skew_tp=None,               # 用户偏好状态转移矩阵
                 # FSMC 相关参数：信道离散状态
                 gamma_levels=[0.2, 0.6, 1.0], # 不同SINR等级
                 gamma_tp=None,               # SINR状态转移矩阵
                 # 正态分布参数，分配层概率
                 mu=2.0,       # 主导层 (k索引) 的均值
                 sigma=0.8,    # 正态分布std
                 layer_sizes=None,   # 每个 (f,k) 的分层大小 (比特)
                 alpha_user=None,    # 每个用户对推荐的接受度
                 seed=123
                 ):
        """
        Parameters:
            M, N, F, K, C_cache, C_rec: 同论文定义
            D_max, phi: 偏好偏离度允许阈值及其惩罚系数
            B: 无线带宽
            BH_B: 回程带宽
            P_m: 每个基站的发送功率 (假设相同)
            w_BH: 回程能耗因子 (与缓存时的能量消耗也可共用)
            skew_states: 用户偏好马尔可夫链各状态 (不同Zipf斜率)
            skew_tp: 用户偏好状态转移矩阵 (H x H)
            gamma_levels: 离散化SINR等级
            gamma_tp: 信道状态转移矩阵 (L x L)
            mu, sigma: 正态分布参数，用于对K层的请求概率分配
            layer_sizes: 若为 None，则随机生成 (F,K) 大小的层比特数
            alpha_user: shape=(N,)，每个用户的推荐接受度 alpha_n
            seed: 随机种子
        """
        self.M = M
        self.N = N
        self.F = F
        self.K = K
        self.C_cache = C_cache
        self.C_rec = C_rec
        self.D_max = D_max
        self.phi = phi
        self.B = B
        self.BH_B = BH_B
        self.P_m = P_m
        self.w_BH = w_BH
        self.mu = mu
        self.sigma = sigma

        rng = np.random.RandomState(seed)

        # 若未传入 alpha_user，则默认为 0.5
        if alpha_user is None:
            alpha_user = np.full(N, 0.5, dtype=float)
        self.alpha_user = alpha_user

        # ============= 用户偏好 Markov 链 =============
        self.skew_states = skew_states
        self.H = len(skew_states)  # 偏好状态数量
        # 如果未提供转移矩阵，则默认：对角线 0.8，其余 0.2/(H-1)
        if skew_tp is None:
            stp = 0.8
            self.skew_tp = np.full((self.H, self.H), (1.0 - stp) / (self.H - 1))
            np.fill_diagonal(self.skew_tp, stp)
        else:
            self.skew_tp = np.array(skew_tp, dtype=float)
        # 每个用户独立地处于 [0..H-1] 中某个状态
        self.user_pref_state = rng.randint(0, self.H, size=N)

        # ============= 信道状态 Markov 链 =============
        self.gamma_levels = gamma_levels
        self.L = len(gamma_levels)  # 离散信道等级数
        if gamma_tp is None:
            stp = 0.8
            self.gamma_tp = np.full((self.L, self.L), (1.0 - stp) / (self.L - 1))
            np.fill_diagonal(self.gamma_tp, stp)
        else:
            self.gamma_tp = np.array(gamma_tp, dtype=float)

        # 对于每个 (m,n) 链路存储一个信道状态索引
        self.channel_state = rng.randint(0, self.L, size=(M, N))

        # ============= 初始化每层大小 layer_sizes =============
        if layer_sizes is None:
            layer_sizes = np.zeros((F, K), dtype=float)
            for f in range(F):
                # 例如基础层在 2~5 Mbits 之间
                base_size = rng.randint(2_000_000, 5_000_000)
                for k_ in range(K):
                    # 每向上一层大小减半等比
                    ratio = 1.0 / (2**k_)
                    layer_sizes[f, k_] = base_size * ratio
        self.layer_sizes = layer_sizes

        # 初始化用户偏好分布 p_{n,f,k}
        self.p = np.zeros((N, F, K), dtype=float)
        self.update_all_users_preference()

        # 缓存状态: X[m,f,k] 0/1
        self.cache_state = np.zeros((M, F, K), dtype=int)

        self.rng = rng
        self.t = 0

    def reset(self):
        """
        重置环境，通常在每个episode开始时调用
        返回初始状态向量 (flatten)
        """
        # 随机用户偏好状态
        self.user_pref_state = self.rng.randint(0, self.H, size=self.N)
        self.update_all_users_preference()

        # 随机信道状态
        self.channel_state = self.rng.randint(0, self.L, size=(self.M, self.N))

        # 清空缓存
        self.cache_state.fill(0)
        self.t = 0

        return self.get_state()

    def get_state(self):
        """
        返回状态向量：包含 p_{n,f,k} (N*F*K) 和 channel_state (M*N)
        也可扩展包含 user_pref_state(n维)；此处暂不纳入
        """
        p_flat = self.p.flatten()                  # (N*F*K,)
        ch_flat = self.channel_state.flatten()     # (M*N,)
        return np.concatenate([p_flat, ch_flat]).astype(np.float32)

    def update_all_users_preference(self):
        """
        根据每个用户的偏好状态(Zipf斜率) + 正态分布, 计算 p_{n,f,k}.
        p_{n,f} = (f^{-skew}) / sum_{i=1..F}(i^{-skew})
        p_k ~ Normal(k; mu, sigma)
        p_{n,f,k} = p_{n,f} * p_k
        """
        for n in range(self.N):
            state_idx = self.user_pref_state[n]       # 用户 n 当前的偏好状态
            skew = self.skew_states[state_idx]
            # -- Zipf over F
            ranks = np.arange(1, self.F+1, dtype=float)
            zipf_unnorm = ranks ** (-skew)
            sum_zipf = zipf_unnorm.sum()
            # -- normal over K
            k_arr = np.arange(1, self.K+1, dtype=float)
            gauss = np.exp(-((k_arr - self.mu)**2)/(2*self.sigma**2))
            gauss /= (np.sqrt(2*np.pi) * self.sigma)  # 归一化常数
            sum_gauss = gauss.sum()

            for f_ in range(self.F):
                pf = zipf_unnorm[f_] / sum_zipf
                for k_ in range(self.K):
                    pk = gauss[k_] / sum_gauss
                    self.p[n, f_, k_] = pf * pk

    def update_user_pref_state(self):
        """用户偏好状态马尔可夫转移"""
        for n in range(self.N):
            old_s = self.user_pref_state[n]
            new_s = self.rng.choice(self.H, p=self.skew_tp[old_s])
            self.user_pref_state[n] = new_s

    def update_channel_state(self):
        """信道状态马尔可夫转移"""
        for m in range(self.M):
            for n in range(self.N):
                old_g = self.channel_state[m, n]
                new_g = self.rng.choice(self.L, p=self.gamma_tp[old_g])
                self.channel_state[m, n] = new_g

    def get_sinr(self, m, n):
        """返回 (m,n)链路的SINR值(离散等级)"""
        idx = self.channel_state[m, n]
        return self.gamma_levels[idx]

    def get_wireless_rate(self, m, n, assoc_count):
        """
        计算无线传输速率: R_{m,n} = (B / assoc_count) * ln(1 + sinr_{m,n}).
        若 assoc_count=0 则返回 0.
        """
        if assoc_count < 1:
            return 0.0
        gamma_mn = self.get_sinr(m, n)
        return (self.B / assoc_count) * np.log(1.0 + gamma_mn + 1e-9)

    def get_backhaul_rate(self, m, assoc_count):
        """
        回程带宽分配: R_m^BH = BH_B / assoc_count.
        """
        if assoc_count < 1:
            return 0.0
        return self.BH_B / assoc_count

    def step(self, X, Y, Z):
        """
        X: (M,F,K) 缓存策略(0/1)
        Y: (M,F,K) 推荐策略(0/1)
        Z: (M,N)   用户关联(0/1)
        返回: (next_state, reward, done, info)
        """
        #------------------------
        # 1) 推荐后偏好 tilde_p (公式(7))
        #------------------------
        tilde_p = np.copy(self.p)  # 保留原始 p
        for n in range(self.N):
            m = np.argmax(Z[:, n])  # 用户 n 关联的基站
            rec_pairs = np.argwhere(Y[m] == 1)  # 该基站推荐的 (f,k)
            if len(rec_pairs) > 0:
                L = len(rec_pairs)
                alpha_n = self.alpha_user[n]
                # 被推荐内容
                for (f_, k_) in rec_pairs:
                    tilde_p[n, f_, k_] = alpha_n*(1.0/L) + (1-alpha_n)*self.p[n, f_, k_]
                # 未被推荐内容
                mask_rec = np.zeros((self.F, self.K), dtype=bool)
                for (f_, k_) in rec_pairs:
                    mask_rec[f_, k_] = True
                for f_ in range(self.F):
                    for kk_ in range(self.K):
                        if not mask_rec[f_, kk_]:
                            tilde_p[n, f_, kk_] = (1 - alpha_n)*self.p[n, f_, kk_]
                # 归一化
                ssum = tilde_p[n].sum()
                if abs(ssum - 1.0) > 1e-9:
                    tilde_p[n] /= ssum
            else:
                # 未推荐则不变
                tilde_p[n] = self.p[n]

        #------------------------
        # 2) 缓存能耗 E_cache (公式(5))
        #------------------------
        old_cache = (self.cache_state == 1)
        new_cache = (X == 1) & (~old_cache)
        # 每个新缓存项的大小 * w_BH
        E_cache = np.sum(new_cache * self.layer_sizes[np.newaxis, :, :]) * self.w_BH
        self.cache_state = X.copy()

        #------------------------
        # 3) 内容传输能耗 E_delivery (公式(12)-(15))
        #------------------------
        E_delivery = 0.0
        total_cached_layers = 0.0
        total_layers = 0.0
        delivered_bits = 0.0   # 用于计算能量效率 (传输了多少比特)

        # 统计基站关联用户数，用于无线带宽平分
        assoc_counts = np.sum(Z, axis=1)
        # 统计回程关联 (可更精细统计谁在请求未缓存层, 这里简化用总关联)
        assoc_counts_bh = np.sum(Z, axis=1)

        for n in range(self.N):
            m = np.argmax(Z[:, n])
            user_assoc_count = max(1, assoc_counts[m])
            R_mn = self.get_wireless_rate(m, n, user_assoc_count)
            user_assoc_count_bh = max(1, assoc_counts_bh[m])
            R_bh = self.get_backhaul_rate(m, user_assoc_count_bh)

            for f_ in range(self.F):
                for k_ in range(1, self.K+1):
                    prob_req = tilde_p[n, f_, k_-1]  # 请求视频f_质量k_的概率
                    if prob_req <= 1e-12:
                        continue
                    # 传输 1..k_ 层
                    for i in range(1, k_+1):
                        size_bits = self.layer_sizes[f_, i-1]
                        # 无线传输能耗 E_trans = P_m * (size_bits / R_{m,n})
                        if R_mn > 1e-9:
                            E_trans = self.P_m * (size_bits / R_mn)
                        else:
                            E_trans = self.P_m * (size_bits / 1e-9)

                        cached = (X[m, f_, i-1] == 1)
                        if cached:
                            E_delivery += prob_req * E_trans
                            total_cached_layers += prob_req
                        else:
                            # 回程能耗
                            if R_bh > 1e-9:
                                E_bh = self.w_BH * (size_bits / R_bh)
                            else:
                                E_bh = self.w_BH * (size_bits / 1e-9)
                            E_delivery += prob_req * (E_trans + E_bh)
                        total_layers += prob_req
                        delivered_bits += prob_req * size_bits

        # 总能耗
        E_total = E_cache + E_delivery

        #------------------------
        # 4) 偏好偏离度 D^t (公式(8))
        #------------------------
        diff = tilde_p - self.p
        D = np.mean(np.sum(np.square(diff), axis=(1,2)))  # (1/N)*∑(f,k)[(tilde_p - p)^2]

        #------------------------
        # 5) 能量效率
        #------------------------
        energy_efficiency = 0.0
        if E_total > 1e-9:
            energy_efficiency = delivered_bits / E_total

        #------------------------
        # 6) 即时回报 (最小化能耗 + 超过偏好阈值惩罚)
        #------------------------
        if D <= self.D_max:
            reward = -E_total
        else:
            reward = -E_total - self.phi * (D - self.D_max)

        #------------------------
        # 7) 收集信息指标
        #------------------------
        cache_hit_rate = 0.0
        if total_layers > 1e-9:
            cache_hit_rate = total_cached_layers / total_layers
        info = {
            "E_total": E_total,
            "E_cache": E_cache,
            "E_delivery": E_delivery,
            "D": D,
            "cache_hit_rate": cache_hit_rate,
            "energy_efficiency": energy_efficiency,
        }

        #------------------------
        # 8) 更新到下一个时隙:
        #    先更新用户偏好状态 + 计算新的 p，再更新信道状态
        #------------------------
        self.update_user_pref_state()
        self.update_all_users_preference()
        self.update_channel_state()
        self.t += 1

        next_state = self.get_state()
        done = False  # 由外部决定episode何时结束
        return next_state, reward, done, info
