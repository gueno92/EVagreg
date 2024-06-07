import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt
import random
class EV_BATTERY:
    
    def __init__(self, P_charg: float,
                capacity: float,
                P_dischar: float,
                eta : float,
                n_EV : int) -> None:
        """
        Initialize the battery
        Args :

        P_charg (float) : max power to charge
        capacity (float) : battery storage capacity
        P_dischar (float) : discharge capacity
        """
        self.P_charg = P_charg
        self.capacity = capacity
        self.P_dischar = P_dischar
        self.eta = eta
        self.n_EV = n_EV


    def read_spot_price(self, path_to_price : str):
        """
        Read the df of price
        """
        df_price = pd.read_csv(path_to_price)

        self.price_ls = df_price['Prix (€/MWh)']

        self.T = len(self.price_ls)



    def generate_contiguous_list(self):
        """
        Generates a list with contiguous blocks of 1s and 0s where the series of 1s starts at a random position and has a random length.
        
        Parameters:
        length (int): Length of the list.
        
        Returns:
        list: The generated list.
        """
        length = self.T
        if length <= 1:
            raise ValueError("Length of the list must be greater than 1.")
        
        # Determine the length of the series of 1s
        len_ones = random.randint(1, length - 1)
        
        # Determine the starting position for the series of 1s
        start_pos = random.randint(0, length - len_ones)
        
        # Generate the list with contiguous blocks of 1s and 0s
        ls = [0] * length
        for i in range(len_ones):
            ls[start_pos + i] = 1
        
        return ls

    def generate_contiguous_matrix(self):
        """
        Generates a matrix with contiguous blocks of 1s and 0s in each column.
        
        Parameters:
        rows (int): Number of rows in the matrix.
        cols (int): Number of columns in the matrix.
        
        Returns:
        np.ndarray: The generated matrix.
        """
        rows = self.T
        cols = self.n_EV
        matrix = np.zeros((rows, cols), dtype=int)
        
        for col in range(cols):
            matrix[:, col] = self.generate_contiguous_list()
        
        self.schedule_matrix = matrix



    def battery_dispatch(self):
        """
        Optimize the battery dispatch using the price collected
        from RTE
        """
        #Time over which to dispatch
        T = len(self.price_ls)

        # Model declaration
        model = Model("battery")


        # Decision variables
        x_ch = model.addVars(T,self.n_EV, lb=0.0, vtype=GRB.CONTINUOUS, name="Charge") # Charge from the battery 
        x_dch = model.addVars(T,self.n_EV, lb=0.0, vtype=GRB.CONTINUOUS, name="Discharge") # Discharge from the battery
        x_soc = model.addVars(T, self.n_EV, lb=0.0, ub = 1, vtype=GRB.CONTINUOUS, name="SOC") # State of charge of battery
        x_dispatch = model.addVars(T, self.n_EV, vtype=GRB.BINARY, name="dispatch") # Decision on dispatch (0 => no dispatch, 1 => dispatch)


        # Objective function: Maximize the revenue of the battery
        model.setObjective(quicksum(quicksum((x_dch[i,k] - x_ch[i,k])*self.price_ls[i] 
                                             for i in range(T)) 
                                             for k in range(self.n_EV)), GRB.MAXIMIZE)


        # Constraints (1)
        # Cannot discharge more than the available energy
        for t in range(T):
            for k in range(self.n_EV):
                model.addConstr(x_dch[t,k] <= x_dispatch[t,k]*x_soc[t,k]*self.capacity * self.schedule_matrix[t,k])
                model.addConstr(x_dch[t,k] <= x_dispatch[t,k]*self.P_dischar * self.schedule_matrix[t,k])
        

        # Constraints (3)
        # Cannot charge more than available and cannot charge while dispatching
        for t in range(T): 
            for k in range(self.n_EV):
                model.addConstr(x_ch[t,k] <= (1-x_dispatch[t,k])*(1-x_soc[t,k])*self.capacity * self.schedule_matrix[t,k])
                model.addConstr(x_ch[t,k] <= (1-x_dispatch[t,k])*self.P_charg * self.schedule_matrix[t,k])

        # Constraint (3)
        # Create the state of charge variable (start with full)
        # Start with a charged battery
        for k in range(self.n_EV):
            model.addConstr(x_soc[0,k] == np.random.uniform(low=0.0, high=1.0))
            model.addConstr(x_soc[T-1,k] >= 0.95)
        
        # State of charge at each time step
        for t in range(1, T):
            for k in range(self.n_EV):
                model.addConstr((x_soc[t,k] - x_soc[t-1,k])*self.capacity == x_ch[t-1,k] - x_dch[t-1,k])


        #Solve the model
        model.optimize()

        # Electrolizer state
        charge = np.zeros((T,self.n_EV))
        
        #Power of each electrolizer
        discharge = np.zeros((T,self.n_EV))
        SOC = np.zeros((T,self.n_EV))

        # Save the variables
        for j in range(T):
            for k in range(self.n_EV):
                charge[j,k] = x_ch[j,k].X
                discharge[j,k] = x_dch[j,k].X
                SOC[j,k] = x_soc[j,k].X


        return charge, discharge, SOC