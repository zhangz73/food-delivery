import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

w = 1
c = 1
R = 10
beta = 0.99
N_RESTAURANTS = 60
N_CUSTOMERS = 150
MAX_BATCH = 2
RESTAURANT_TO_BUSINESSZONE = {}
CUSTOMER_TO_BUSINESSZONE = {}

def generate_locations(seed = None):
    np.random.seed(seed)
    restaurants = np.random.uniform(size = (N_RESTAURANTS, 2))
    customers = np.random.uniform(size = (N_CUSTOMERS, 2))
    dist_matrix = np.linalg.norm(restaurants[:, None, :] - customers[None, :, :], axis = 2)
    tau = np.zeros((dist_matrix.shape[0], dist_matrix.shape[1], MAX_BATCH))
    batch_cost = 0.1
    for batch in range(MAX_BATCH):
        tau[:,:,batch] = dist_matrix + batch_cost * batch
    return tau

def in_same_zone(restaurant, customer):
    r_zone, c_zone = -1, -1
    if restaurant in RESTAURANT_TO_BUSINESSZONE:
        r_zone = RESTAURANT_TO_BUSINESSZONE[restaurant]
    if customer in CUSTOMER_TO_BUSINESSZONE:
        c_zone = CUSTOMER_TO_BUSINESSZONE[customer]
    return r_zone == c_zone

tau = generate_locations(seed = 123) #np.ones((N_RESTAURANTS, N_CUSTOMERS))
LAMBDA = np.ones((N_RESTAURANTS, N_CUSTOMERS))
MU_P = 0.5 * N_RESTAURANTS * N_CUSTOMERS
parttime_cost_ub = np.sum(np.max(tau, axis = 2) * LAMBDA * w * 2)
V_bar_ub = w / np.log(1/beta)
lambda_ub = np.max(LAMBDA)
parttime_max_reposition_distance = np.inf

model = gp.Model()
#model.setParam("Method", 2)
#model.setParam("NodeMethod", 2)
model.setParam("NonConvex", 2)
#model.setParam("OptimalityTol", 1e-4)
#model.setParam("FeasibilityTol", 1e-4)

lam_f_rcb = model.addVars(N_RESTAURANTS, N_CUSTOMERS, MAX_BATCH, lb = 0, ub = lambda_ub, vtype = GRB.CONTINUOUS, name = "lambda_f_rc")
lam_f_cr = model.addVars(N_CUSTOMERS, N_RESTAURANTS, lb = 0, ub = lambda_ub, vtype = GRB.CONTINUOUS, name = "lambda_f_cr")
lam_p_rcb = model.addVars(N_RESTAURANTS, N_CUSTOMERS, MAX_BATCH, lb = 0, ub = lambda_ub, vtype = GRB.CONTINUOUS, name = "lambda_p_rc")
lam_p_cr = model.addVars(N_CUSTOMERS, N_RESTAURANTS, lb = 0, ub = lambda_ub, vtype = GRB.CONTINUOUS, name = "lambda_p_cr")
mu_p_in = model.addMVar(N_RESTAURANTS, lb = 0, ub = MU_P, vtype = GRB.CONTINUOUS, name = "mu_p_in")
mu_p_out = model.addMVar(N_CUSTOMERS, lb = 0, ub = MU_P, vtype = GRB.CONTINUOUS, name = "mu_p_out")
V_r = model.addMVar(N_RESTAURANTS, lb = R, ub = V_bar_ub, vtype = GRB.CONTINUOUS, name = "V_r")
V_c = model.addMVar(N_CUSTOMERS, lb = R, ub = V_bar_ub, vtype = GRB.CONTINUOUS, name = "V_c")
V_bar = model.addVar(lb = R, ub = V_bar_ub, vtype = GRB.CONTINUOUS, name = "V_bar")
total_parttime_cost = model.addVar(lb = 0, ub = parttime_cost_ub, vtype = GRB.CONTINUOUS, name = "parttime_cost")

## Add constraints
for i in range(N_RESTAURANTS):
    model.addConstr(V_bar >= V_r[i])
    model.addConstr(mu_p_in[i] * (V_bar - V_r[i]) == 0)
    model.addConstr(mu_p_in[i] + gp.quicksum(lam_p_cr[j, i] for j in range(N_CUSTOMERS)) == gp.quicksum(lam_p_rcb[i, j, b] for j in range(N_CUSTOMERS) for b in range(MAX_BATCH)))
    model.addConstr(gp.quicksum(lam_f_cr[j, i] for j in range(N_CUSTOMERS)) == gp.quicksum(lam_f_rcb[i, j, b] for j in range(N_CUSTOMERS) for b in range(MAX_BATCH)))

for j in range(N_CUSTOMERS):
    model.addConstr(mu_p_out[j] * (V_c[j] - R) == 0)
    model.addConstr(mu_p_out[j] + gp.quicksum(lam_p_cr[j, i] for i in range(N_RESTAURANTS)) == gp.quicksum(lam_p_rcb[i, j, b] for i in range(N_RESTAURANTS) for b in range(MAX_BATCH)))
    model.addConstr(gp.quicksum(lam_f_rcb[i, j, b] for i in range(N_RESTAURANTS) for b in range(MAX_BATCH)) == gp.quicksum(lam_f_cr[j, i] for i in range(N_RESTAURANTS)))

for i in range(N_RESTAURANTS):
    for j in range(N_CUSTOMERS):
        model.addConstr(V_c[j] >= -c * tau[i, j, 0] + beta ** tau[i, j, 0] * V_r[i])
        for b in range(MAX_BATCH):
            model.addConstr(V_r[i] >= -c * tau[i, j, b] + beta ** tau[i, j, b] * V_c[j])
        model.addConstr(lam_p_cr[j,i] * (V_c[j] + c * tau[i, j, 0] - beta ** tau[i, j, 0] * V_r[i]) == 0)
        model.addConstr(gp.quicksum((lam_f_rcb[i, j, b] + lam_p_rcb[i, j, b]) * (b + 1) for b in range(MAX_BATCH)) == LAMBDA[i, j])
        ## Add upper bound on part-time price
#        model.addConstr(V_r[i] + c * tau[i, j] - beta ** tau[i, j] * V_c[j] <= w * (tau[i, j] + np.max(tau)))
        ## Restrict full-time to travel within each business zone (i.e. da_id)
        if not in_same_zone(i, j):
            for b in range(MAX_BATCH):
                model.addConstr(lam_f_rcb[i, j, b] == 0)
            model.addConstr(lam_f_cr[j, i] == 0)
        ## Restrict part-time to reposition within some distance threshold
        if tau[i, j, 0] > parttime_max_reposition_distance:
            model.addConstr(lam_p_cr[j, i] == 0)

model.addConstr(MU_P >= gp.quicksum(mu_p_in[i] for i in range(N_RESTAURANTS)))
const_parttime_cost = model.addConstr(gp.quicksum(lam_p_rcb[i, j, b] * (V_r[i] + c * tau[i, j, b] - beta ** tau[i, j, b] * V_c[j]) for i in range(N_RESTAURANTS) for j in range(N_CUSTOMERS) for b in range(MAX_BATCH)) == total_parttime_cost)

## Add objective function
obj_func = gp.quicksum(w * lam_f_rcb[i, j, b] * tau[i, j, b] for i in range(N_RESTAURANTS) for j in range(N_CUSTOMERS) for b in range(MAX_BATCH)) + gp.quicksum(w * lam_f_cr[j, i] * tau[i, j, 0] for i in range(N_RESTAURANTS) for j in range(N_CUSTOMERS)) + total_parttime_cost
model.setObjective(obj_func, GRB.MINIMIZE)
model.optimize()
obj_val = model.ObjVal
print(obj_val)

print(total_parttime_cost)
