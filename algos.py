import numpy as np
import os

#beta1,beta2,beta3,beta4,beta5
BOUNDS = [(1e3,1e6),(1,100),(1e-3,1e-1),(0.1,3),(-120,-40)]
LB = np.array([b[0] for b in BOUNDS]) #lower bound
UB = np.array([b[1] for b in BOUNDS]) #upper bound
DIFF = UB - LB

#loading the data
try:
    train_data = np.loadtxt('radar_train.txt')
    test_data = np.loadtxt('radar_test.txt')
except OSError:
    print("No data found")
    exit()

# for the parts of the PW func
def precompute_features(data):
    R = data[:, 0]
    theta_rad = np.radians(data[:, 1])
    T = data[:, 2]
    P = data[:, 3]
    y = data[:, 4]
    X1 = (np.cos(theta_rad)**2) / (R**4)
    X2 = T - 15.0
    X3 = np.log(P)
    return X1, X2, X3, y

X1_tr, X2_tr, X3_tr, y_tr = precompute_features(train_data)
X1_te, X2_te, X3_te, y_te = precompute_features(test_data)

class BudgetTracker:

    def __init__(self, max_cost=100000):
        self.max_cost = max_cost
        self.cost = 0
        self.best_mse = float('inf')
        self.best_w = None
        self.stop_flag = False
        self.last_hit_cost = 0
    
    def reset(self):
        self.cost = 0
        self.best_mse = float('inf')
        self.best_w = None
        self.stop_flag = False
        self.last_hit_cost = 0

    def check(self, cost_increment):
        if self.cost + cost_increment > self.max_cost:
            self.stop_flag = True
            return False
        return True

    def update(self, w, mse, cost_increment):
        self.cost += cost_increment
        if mse < self.best_mse:
            self.best_mse = mse
            self.best_w = w.copy()
            self.last_hit_cost = self.cost
        if self.cost >= self.max_cost:
            self.stop_flag = True

tracker = BudgetTracker(max_cost=100000)

# to get back to the normal bounds
def get_beta(w):
    return LB + w * DIFF
    
def model_predict(beta, X1, X2, X3):
    # gia overflow
    exp_val = np.clip(-beta[2]*X2, -100, 100) 
    return beta[0]*X1 + beta[1]*np.exp(exp_val) + beta[3]*X3 + beta[4]
    
def func(w):
    # check if we can afford it
    if not tracker.check(1): 
        return 1e9
    
    # check bounds
    if np.any(w < 0) or np.any(w > 1):
        tracker.cost += 1 
        return 1e9
    # get real values and calc mse 
    beta = get_beta(w)
    y_pred = model_predict(beta, X1_tr, X2_tr, X3_tr)
    mse = np.mean((y_pred - y_tr)**2)
    
    tracker.update(w, mse, 1)
    return mse
    
def gradient(w):
    # check if we can afford it
    if not tracker.check(5): 
        return np.zeros_like(w)
    
    # check the bounds
    if np.any(w < 0) or np.any(w > 1):
        tracker.cost += 5
        return np.ones_like(w) * 1e5

    beta = get_beta(w)

    y_pred = model_predict(beta, X1_tr, X2_tr, X3_tr)
    residual = y_pred - y_tr
    N = len(y_tr)
    
    d_beta1 = X1_tr
    exp_term = np.exp(np.clip(-beta[2] * X2_tr, -100, 100))
    d_beta2 = exp_term
    d_beta3 = beta[1] * exp_term * (-X2_tr)
    d_beta4 = X3_tr
    d_beta5 = np.ones_like(y_pred)
    
    grad_beta = np.zeros(5)
    factor = 2/N * residual
    grad_beta[0] = np.sum(factor * d_beta1)
    grad_beta[1] = np.sum(factor * d_beta2)
    grad_beta[2] = np.sum(factor * d_beta3)
    grad_beta[3] = np.sum(factor * d_beta4)
    grad_beta[4] = np.sum(factor * d_beta5)
    
    tracker.update(w, tracker.best_mse, 5) # update cost but keep old mse
    return grad_beta * DIFF
    
def hessian(w):
    # check if we can afford it, return the I if not
    if not tracker.check(15): return np.eye(len(w))
    
    tracker.cost += 15
    eps = 1e-5
    n = len(w)
    H = np.zeros((n, n))
    
    
    # same as original gradient, without the tracker
    def raw_grad(wk):
        beta = get_beta(wk)
        y_p = model_predict(beta, X1_tr, X2_tr, X3_tr)
        res = y_p - y_tr
        fac = (2/len(y_tr)) * res
        exp_t = np.exp(np.clip(-beta[2] * X2_tr, -100, 100))
        gb = np.zeros(5)
        gb[0] = np.sum(fac * X1_tr)
        gb[1] = np.sum(fac * exp_t)
        gb[2] = np.sum(fac * (beta[1] * exp_t * (-X2_tr)))
        gb[3] = np.sum(fac * X3_tr)
        gb[4] = np.sum(fac * 1.0)
        return gb * DIFF

    #calculations
    g0 = raw_grad(w)
    for i in range(n):
        w_eps = w.copy()
        w_eps[i] += eps
        g_eps = raw_grad(w_eps)
        H[:, i] = (g_eps - g0) / eps
    
    
    return 0.5 * (H + H.T)
    
def newton_dogleg(x0):
    x = x0.copy()
    radius = 0.1 # initial radius
    
    # gia oso yparxei budget
    while not tracker.stop_flag:
        g = gradient(x)
        if tracker.stop_flag: break # teleiose to budget meta ton ypol tis gradient
        
        H = hessian(x)
        if tracker.stop_flag: break # teleiose to budget meta ton ypol tou hessian
        
        if np.linalg.norm(g) < 1e-4: break # an h gradient poly mikri, telos
        
        # cauchy point
        denom = np.dot(g.T, np.dot(H, g))
        if denom <= 0:
            tau = 1.0
        else:
            tau = min(1.0, np.linalg.norm(g)**3 / (radius * denom))
        pC = -tau * (radius / (np.linalg.norm(g)+1e-10)) * g

        # newton point
        try:
            pN = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            pN = pC

        # choose p
        norm_pN = np.linalg.norm(pN)
        if norm_pN <= radius:
            p = pN
        elif np.linalg.norm(pC) >= radius:
            p = (radius / np.linalg.norm(pC)) * pC
        else:
            p_diff = pN - pC
            a = np.dot(p_diff, p_diff)
            b = 2 * np.dot(pC, p_diff)
            c = np.dot(pC, pC) - radius**2
            discriminant = b**2 - 4*a*c
            if discriminant < 0: beta = 0
            else: beta = (-b + np.sqrt(discriminant)) / (2*a)
            p = pC + beta * p_diff
        
        x_new = np.clip(x + p, 0, 1)
        
        # check reduction
        f_curr = tracker.best_mse # exoume to fx apo prin
        f_new = func(x_new)
        if tracker.stop_flag: break
        
        # update tin aktina
        act_red = f_curr - f_new
        pred_red = -(np.dot(g, p) + 0.5 * np.dot(p, np.dot(H, p)))
        
        rho = act_red / (pred_red + 1e-10)
        
        if rho > 0.2:
            x = x_new
            if rho > 0.75:
                radius = min(2.0, 2*radius)
        else:
            radius = 0.5 * radius
            
        if radius < 1e-6: break
    return tracker.best_w
    
def line_search_wolfe(w, p, g, f_curr, func_obj, grad_obj, c1=1e-4, c2=0.9):

    alpha = 1.0 # initial a
    alpha_max = 10.0
    alpha_prev = 0.0
    f_prev = f_curr
    
    g_dot_p = np.dot(g, p) # paragogos kateythinsis
    
    # zoom func searches in [a_lo, a_hi]
    def zoom(a_lo, a_hi, f_lo):
        for _ in range(20): # safety break
            # interpolation-bisection 
            a_j = 0.5 * (a_lo + a_hi)
            
            w_aj = np.clip(w + a_j * p, 0, 1)
            f_aj = func_obj(w_aj)
            
            #  armijo condition 1
            # sigrisi me f_curr (f(0))
            if (f_aj > f_curr + c1 * a_j * g_dot_p) or (f_aj >= f_lo):
                a_hi = a_j
            else:
                # curvature condition 2 - strong wolfe
                g_aj = grad_obj(w_aj)
                g_aj_dot_p = np.dot(g_aj, p)
                
                if abs(g_aj_dot_p) <= -c2 * g_dot_p:
                    return a_j # success
                
                if g_aj_dot_p * (a_hi - a_lo) >= 0:
                    a_hi = a_lo
                
                a_lo = a_j
                f_lo = f_aj
                
        return a_lo # gia fallback
    # main loop
    for i in range(20): # max iters
        w_new = np.clip(w + alpha * p, 0, 1)
        f_new = func_obj(w_new)
        
        # armijo
        if (f_new > f_curr + c1 * alpha * g_dot_p) or (i > 0 and f_new >= f_prev):
            # if failed check between previous and current
            return zoom(alpha_prev, alpha, f_prev)
        
        # curvature
        g_new = grad_obj(w_new)
        g_new_dot_p = np.dot(g_new, p)
        
        if abs(g_new_dot_p) <= -c2 * g_dot_p:
            return alpha # ikanopoiountai kai oi dyo
        
        if g_new_dot_p >= 0:
            return zoom(alpha, alpha_prev, f_new)
        
        # ayxhse bima
        alpha_prev = alpha
        f_prev = f_new
        alpha = min(2.0 * alpha, alpha_max)
        
    return alpha 
    
#  BFGS  
def bfgs(x0):
    x = x0.copy()
    n = len(x)
    I = np.eye(n)
    Bk = I
    g = gradient(x)
    
    # arxiki timi synartisis
    f_curr = func(x) 
    
    while not tracker.stop_flag and np.linalg.norm(g) > 1e-4:
        # kateythinsi
        p = -np.dot(Bk, g)
        
        # line search 
        alpha = line_search_wolfe(x, p, g, f_curr, func, gradient)
        
        if tracker.stop_flag: break
        
        # update to simeio
        x_new = np.clip(x + alpha * p, 0, 1)
        f_new = func(x_new)
        
        # enimerosi ton metavliton
        s = x_new - x
        g_new = gradient(x_new)
        y = g_new - g
        
        if tracker.stop_flag: break
        
        # update ton Bk
        rho_denom = np.dot(y, s)
        if rho_denom > 1e-10: # gia na min krasarei an einai poly mikro,dld sxedon 0
            rho = 1.0 / rho_denom
            V = I - rho * np.outer(s, y)
            Bk = np.dot(V, np.dot(Bk, V.T)) + rho * np.outer(s, s)
        
        # enimerosi metavliton
        x = x_new
        g = g_new
        f_curr = f_new
        
    return tracker.best_w
    
# nelder mead
def nelder_mead(x0):
    dim = len(x0)
    
    # arxikopoihsi tou simplex 
    simplex = [x0]
    for i in range(dim):
        pt = x0.copy()
        pt[i] = pt[i] + 0.05 if pt[i] < 0.5 else pt[i] - 0.05
        simplex.append(pt)
    
    simplex = np.array(simplex)
    f_vals = np.array([func(p) for p in simplex])
    
    # parametroi, oi default apo diafaneies
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho   = 0.5  # contraction
    sigma = 0.5  # shrink
    
    while not tracker.stop_flag:
        # sort best to worse 
        order = np.argsort(f_vals)
        simplex = simplex[order]
        f_vals = f_vals[order]
        
        best = simplex[0]
        worst = simplex[-1]
        second_worst = f_vals[-2]
        
        # ypologismos tou kentrou
        centroid = np.mean(simplex[:-1], axis=0)
        
        # reflection
        xr = centroid + alpha * (centroid - worst)
        xr = np.clip(xr, 0, 1) # bounds handling
        fr = func(xr)
        if tracker.stop_flag: break
        
        # proti periptosi
        if fr < second_worst and fr >= f_vals[0]:
            simplex[-1] = xr
            f_vals[-1] = fr
            
        # expansion
        elif fr < f_vals[0]:
            xe = centroid + gamma * (xr - centroid)
            xe = np.clip(xe, 0, 1)
            fe = func(xe)
            if tracker.stop_flag: break
            
            if fe < fr:
                simplex[-1] = xe
                f_vals[-1] = fe
            else:
                simplex[-1] = xr
                f_vals[-1] = fr
            
        # contraction
        else:
            perform_shrink = False
            
            # external contraction
            if fr < f_vals[-1] and fr >= second_worst:
                xc = centroid + rho * (xr - centroid)
                xc = np.clip(xc, 0, 1)
                fc = func(xc)
                if tracker.stop_flag: break
                
                if fc <= fr:
                    simplex[-1] = xc
                    f_vals[-1] = fc
                else:
                    perform_shrink = True
            
            # internal contraction 
            else:
                xc = centroid + rho * (worst - centroid)
                xc = np.clip(xc, 0, 1)
                fc = func(xc)
                if tracker.stop_flag: break
                
                if fc < f_vals[-1]:
                    simplex[-1] = xc
                    f_vals[-1] = fc
                else:
                    perform_shrink = True
            
            # shrink
            if perform_shrink:
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], 0, 1)
                    f_vals[i] = func(simplex[i])
                    if tracker.stop_flag: break
                    
    return tracker.best_w
    
# genetic algorithm
def ga(x0,pop_size=30, mutation_rate=0.1):
    dim = 5
    pop = np.random.rand(pop_size, dim)
    
    pop[0] = x0.copy()
    # calculating the fitness
    fitness = []
    for ind in pop:
        if not tracker.check(1): break 
        fitness.append(func(ind)) 
    
    # mporei na teleiose to budget stin arxikopoisi
    if len(fitness) < pop_size: return tracker.best_w if tracker.best_w is not None else pop[0]
    
    fitness = np.array(fitness)
    
    while not tracker.stop_flag:
        new_pop = []
        # elitismos, krata to kalytero
        best_idx = np.argmin(fitness)
        new_pop.append(pop[best_idx].copy())
        
        while len(new_pop) < pop_size:
            if tracker.stop_flag: break
            
            # tournament selection
            i1, i2 = np.random.choice(pop_size, 2)
            p1 = pop[i1] if fitness[i1] < fitness[i2] else pop[i2]
            i1, i2 = np.random.choice(pop_size, 2)
            p2 = pop[i1] if fitness[i1] < fitness[i2] else pop[i2]
            
            # crossover arithmitiko
            if np.random.rand() < 0.8: # crossover pithanotita 0.8
                alpha = np.random.rand()
                child = alpha*p1 + (1-alpha)*p2
            else:
                child = p1.copy()
                
            # mutation
            for d in range(dim):
                if np.random.rand() < mutation_rate:
                    child[d] += np.random.normal(0, 0.05)
                
            #gia times ektos orion    
            child = np.clip(child, 0, 1)
            new_pop.append(child)
            
        pop = np.array(new_pop)[:pop_size]
        
        # evaluation of new population
        for i in range(len(pop)):
            if tracker.stop_flag: break
            # ton best ton exoume idi, ara glitonoume kostos
            if i == 0: 
                fitness[i] = fitness[best_idx]
            else:
                fitness[i] = func(pop[i])
            
    return tracker.best_w
    
#pso
def pso(x0, pop_size=30):
    dim = 5
    
    # parameters (opos stin theoria)
    phi1 = 2.05
    phi2 = 2.05
    phi = phi1 + phi2
    # to x
    chi = 2.0 / np.abs(2.0 - phi - np.sqrt(phi**2 - 4*phi)) # peripou 0.729
    
    # initialization
    X = np.random.rand(pop_size, dim)
    if x0 is not None:
        X[0] = x0.copy()
    
    # initialize the velocities
    V = np.zeros((pop_size, dim)) 
    
    P_best = X.copy()
    P_best_val = []
    
    # proti axiologisi
    for x in X:
        if not tracker.check(1): break
        P_best_val.append(func(x))
    
    if len(P_best_val) < pop_size: return tracker.best_w if tracker.best_w is not None else X[0]
    P_best_val = np.array(P_best_val)
    
    g_best_idx = np.argmin(P_best_val)
    g_best = P_best[g_best_idx].copy()
    g_best_val = P_best_val[g_best_idx]
    
    while not tracker.stop_flag:
        r1 = np.random.rand(pop_size, dim)
        r2 = np.random.rand(pop_size, dim)
        
        # h exisosi taxythtas
        V = chi * (V + phi1 * r1 * (P_best - X) + phi2 * r2 * (g_best - X))
        
        # enimerosi thesis kai clip gia na min xefygoun
        X = np.clip(X + V, 0, 1)
        
        # nea axiologisi kai enimerosi 
        for i in range(pop_size):
            if tracker.stop_flag: break
            val = func(X[i])
            
            if val < P_best_val[i]:
                P_best_val[i] = val
                P_best[i] = X[i].copy()
                if val < g_best_val:
                    g_best_val = val
                    g_best = X[i].copy()
                    
    return tracker.best_w

# tuning 
def run_tuning(points):
    print("\n--- Φάση Προεπεξεργασίας (Parameter Tuning) ---")
    print("Δοκιμή παραμέτρων σε μικρά πειράματα (Budget=10000)...")
    
    n_runs = len(points)
    
    # gia tin ga
    ga_configs = [
        {'pop': 20, 'mut': 0.05},
        {'pop': 50, 'mut': 0.10},
        {'pop': 70, 'mut': 0.15},
        {'pop': 100, 'mut': 0.20}
    ]
    best_ga_mse = float('inf')
    best_ga_params = ga_configs[0]
    
    for cfg in ga_configs:
        mses = []
        for i in range(n_runs): # 30 iterations 
            tracker.reset()
            tracker.max_cost = 10000
             
            x0 = points[i]
            ga(x0, pop_size=cfg['pop'], mutation_rate=cfg['mut'])
            
            mses.append(tracker.best_mse)
        
        avg_mse = np.mean(mses)
        print(f"GA Config {cfg}: Mean MSE = {avg_mse:.5f}")
        if avg_mse < best_ga_mse:
            best_ga_mse = avg_mse
            best_ga_params = cfg
            
    print(f"Επιλεγμένες παράμετροι GA: {best_ga_params}")

    # gia tin pso 
    pso_configs = [
        {'pop': 20},
        {'pop': 40},
        {'pop': 60},
        {'pop': 80}
    ]
    best_pso_mse = float('inf')
    best_pso_params = pso_configs[0]
    
    for cfg in pso_configs:
        mses = []
        for i in range(n_runs):
        
            tracker.reset()
            tracker.max_cost = 10000
            
            x0 = points[i]
            pso(x0, pop_size=cfg['pop'])
            
            mses.append(tracker.best_mse)
            
        avg_mse = np.mean(mses)
        print(f"PSO Config {cfg}: Mean MSE = {avg_mse:.5f}")
        if avg_mse < best_pso_mse:
            best_pso_mse = avg_mse
            best_pso_params = cfg
            
    print(f"Επιλεγμένες παράμετροι PSO: {best_pso_params}")
    
    return best_ga_params, best_pso_params
    
# main execution
if __name__ == "__main__":

    N_RUNS = 30
    np.random.seed(3) # o random seed gia anaparogisimotita
    
    if os.path.exists("initial_points.txt"):
        print(f"Found existing initial points file: initial_points.txt")
        print("Loading saved points...")
        initial_points_store = np.loadtxt("initial_points.txt")
    else:
        print(f"File initial_points.txt not found.")
        print("Generating new random initial points and saving...")
        initial_points_store = np.random.rand(N_RUNS, 5)
        # save the points
        np.savetxt("initial_points.txt", initial_points_store, fmt='%.8f')
        print(f"Saved initial points to initial_points.txt")
        
    # tuning tis parametrous
    ga_params, pso_params = run_tuning(initial_points_store)
    
    print("\n--- Main Experiments (Budget=100000, N=30) ---")
    
        
    # oi algorithmoi
    algos = {
        'Newton': newton_dogleg,
        'BFGS': bfgs,
        'NelderMead': nelder_mead,
        'GA': lambda x: ga(x, pop_size=ga_params['pop'], mutation_rate=ga_params['mut']),
        'PSO': lambda x: pso(x, pop_size=pso_params['pop'])
    }
    
    all_results = {k: [] for k in algos.keys()}
    
    for name, algo in algos.items():
        filename = f"{name}.txt"
        print(f"Running {name}, saving to {filename}...")
        
        with open(filename, "w") as f:
            # kefalida gia to arxeio
            f.write("Exper MSEtrain MSEtest LastHit Beta1 Beta2 Beta3 Beta4 Beta5\n")
            
            for i in range(N_RUNS):
                x0 = initial_points_store[i]
                
                tracker.reset()
                tracker.max_cost = 100000  # to budget mas

                algo(x0)
                
                # get results
                best_w = tracker.best_w if tracker.best_w is not None else x0
                mse_train = tracker.best_mse
                
                # these to last hit
                last_hit = tracker.last_hit_cost if tracker.last_hit_cost > 0 else tracker.cost
                
                # ypologismos mse
                beta_final = get_beta(best_w)
                y_pred_test = model_predict(beta_final, X1_te, X2_te, X3_te)
                mse_test = np.mean((y_pred_test - y_te)**2)
                
                all_results[name].append(mse_test)
                
                # eggrafi sto arxeio
                line = (f"{i+1:2d} {mse_train:.6f} {mse_test:.6f} {last_hit:5d} "
                        f"{beta_final[0]:.4f} {beta_final[1]:.4f} {beta_final[2]:.6f} "
                        f"{beta_final[3]:.4f} {beta_final[4]:.4f}\n")
                
                f.write(line)
