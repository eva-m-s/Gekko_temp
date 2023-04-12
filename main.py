from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def plot_network(net):
    g = nx.Graph()
    for i in range(net.shape[0]):
        for j in range(net.shape[1]):
            if net[i, j] == 1:
                g.add_edge('Switch ' + str(i+1), 'Controller ' + str(j+1))
    pos = nx.spring_layout(g)
    nx.draw(g, pos, with_labels=True)
    plt.show()


def sdn_opt():
    # Initialize Model
    m = GEKKO(remote=False)

    # # Define penalty factor
    # penalty_factor = 1000

    # Define the number of switches and controllers
    num_switches = 10
    num_controllers = 3

    # Define the maximum load for each controller
    max_load = [20, 30, 40]

    # Define the loads needed to control each switch
    switch_loads = np.random.randint(1, 10, size=num_switches)

    # Define the connectivity matrix between switches and controllers
    connectivity = np.random.randint(0, 2, size=(num_switches, num_controllers))

    # Define the propagation latency matrix between switches and controllers
    latency = np.random.randint(1, 10, size=(num_switches, num_controllers))

    # Define the decision variables
    # Each element z[i][j] represents whether switch i is assigned to controller j
    # x[j] represents whether controller j is used or not
    # y[j] represents the total load on controller j
    z = m.Array(m.Var, (num_switches, num_controllers), lb=0, ub=1, integer=True)
    x = m.Array(m.Var, (num_controllers,), lb=0, ub=1, integer=True)
    y = m.Array(m.Var, (num_controllers,), lb=0, integer=True)

    # Initialize a binary variable z[i, j]
    for i in range(num_switches):
        for j in range(num_controllers):
            z[i, j] = m.Var(lb=0, ub=1, integer=True)

    # Each switch can only be assigned to one controller
    for i in range(num_switches):
        m.Equation(m.sum(z[i, :]) == 1)

    # The loads on each controller cannot exceed its capacity
    for j in range(num_controllers):
        m.Equation(m.sum([switch_loads[i] * z[i, j] for i in range(num_switches)]) <= y[j])
        m.Equation(y[j] <= max_load[j])

    # Each switch must be assigned to exactly one controller
    for i in range(num_switches):
        for j in range(num_controllers):
            m.Equation(z[i, j] <= x[j])

    # # Define the penalty term
    # penalty = m.sum([penalty_factor * m.if3(y[j] - max_load[j], y[j] - max_load[j], 0) for j in range(num_controllers)])

    # Define the objective function as the maximum propagation latency between switches and controllers
    # obj = m.max2(m.sum([latency[i][j] * z[i, j] * x[j] for i in range(num_switches) for j in range(num_controllers)]), 1)
    # m.Obj(obj + penalty)

    # # Define the objective function as the maximum propagation latency between switches and controllers
    # obj = m.Var()
    # m.Equation(
    #     obj >= m.sum([latency[i][j] * z[i, j] * x[j] for i in range(num_switches) for j in range(num_controllers)]))
    # m.Obj(obj)

    # Define the objective function as the maximum propagation latency between switches and controllers
    obj = m.max2(
        m.sum([latency[i][j] * z[i, j] * x[j] for i in range(num_switches) for j in range(num_controllers)]), 1)
    m.Obj(obj)

    m.options.SOLVER = 1
    m.solve(disp=True)

    #return [x[j].value[0] for j in range(num_controllers)]

    return connectivity

if __name__ == '__main__':
    connectivity = sdn_opt()
    plot_network(connectivity)
