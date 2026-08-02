import numpy as np

class BOCPD:
    """
    Bayesian Online Changepoint Detection for buy_volume_ratio.
    """
    def __init__(self, hazard_rate=0.01, mu0=0.5, kappa0=1.0, alpha0=1.0, beta0=0.01):
        self.hazard = hazard_rate
        self.mu0, self.kappa0, self.alpha0, self.beta0 = mu0, kappa0, alpha0, beta0
        
        self.run_length_probs = np.array([1.0])
        self.params = [(mu0, kappa0, alpha0, beta0)]
        self._changepoint_prob = 0.0
        
    def update(self, bar):
        vol = bar["volume"]
        if vol < 1e-9:
            x = 0.5
        else:
            x = bar["buy_volume"] / vol
            
        pred_probs = []
        for mu, kappa, alpha, beta in self.params:
            var = beta * (kappa + 1) / (alpha * kappa)
            pred_probs.append(self._student_t_pdf(x, 2 * alpha, mu, max(var, 1e-9)))
            
        pred_probs = np.array(pred_probs)
        
        growth = self.run_length_probs * pred_probs * (1 - self.hazard)
        cp = np.sum(self.run_length_probs * pred_probs * self.hazard)
        
        new_probs = np.append(cp, growth)
        sum_probs = new_probs.sum()
        if sum_probs > 0:
            new_probs /= sum_probs
        else:
            new_probs = np.zeros_like(new_probs)
            new_probs[0] = 1.0
            
        self.run_length_probs = new_probs
        
        new_params = [(self.mu0, self.kappa0, self.alpha0, self.beta0)]
        for mu, kappa, alpha, beta in self.params:
            kappa_n = kappa + 1.0
            mu_n = (kappa * mu + x) / kappa_n
            alpha_n = alpha + 0.5
            beta_n = beta + kappa * (x - mu) ** 2 / (2 * kappa_n)
            new_params.append((mu_n, kappa_n, alpha_n, beta_n))
            
        self.params = new_params
        
        if len(self.run_length_probs) > 200:
            self.run_length_probs = self.run_length_probs[:200]
            self.params = self.params[:200]
            sum_probs = self.run_length_probs.sum()
            if sum_probs > 0:
                self.run_length_probs /= sum_probs
                
        self._changepoint_prob = float(cp)
        
    def _student_t_pdf(self, x, df, mu, var):
        return (1.0 + (x - mu)**2 / (df * var)) ** (-(df + 1) / 2)
        
    @property
    def changepoint_probability(self):
        return self._changepoint_prob
