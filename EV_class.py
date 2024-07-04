import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta

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
        np.random.seed(0)
        self.P_charg = P_charg
        self.capacity = capacity
        self.P_dischar = P_dischar
        self.eta = eta
        self.n_EV = n_EV


    def read_spot_price(self, path_to_price : str, 
                        resample = False):
        """
        Read the df of price
        """
        df_price = pd.read_csv(path_to_price)

        self.price_ls = df_price['Prix (€/MWh)']
        self.delta_T =1
        self.time_step = 60
        if resample :
            self.delta_T = 1/6
            self.time_step = 60*self.delta_T
            resampled_list = []
            for elem in self.price_ls:
                resampled_list.extend([elem] * 6)
            self.price_ls = resampled_list

        self.T = len(self.price_ls)
    
    def definec_charger_charac(self,
                               charge_type: str = 'Electra'):
        """
        Method that define the characteristic of the charging
        point

        Args : 
        charge_type (str) : Choose one type in the list 'Electra' , 'Wallbox'
        """

        if charge_type == 'Electra':
            # Ref : https://www.go-electra.com/en/landowner/
            self.P_charg_point = 150
        elif charge_type == 'Wallbox':
            # Ref : https://www.monkitsolaire.fr/blog/meilleure-borne-de-recharge
            self.P_charg_point = 22
        elif charge_type == 'Circontrol':
            # Ref : https://www.monkitsolaire.fr/blog/meilleure-borne-de-recharge
            self.P_charg_point = 7.4
        

    def reformate_charing_time(self,
                               charging_schedule):
        """
        Reformate the charging schedule to make it suitable with 
        the optimization process

        Args : 
        charging_schedule : tuple of datetime object of format "MM/DD HH:mm"
        """
        start_price_date = datetime(2024, 6, 19, 18, 00)
        #Determine the start position for the optimization
        detla_time_start = charging_schedule[0] - start_price_date
        # for now the steps are of 10 minutes
        start_pos = int(detla_time_start.total_seconds()/(60*self.time_step))


        # Determine the charging duration 
        detla_time_charge = charging_schedule[1] -  charging_schedule[0]
        duration_charging = int(detla_time_charge.total_seconds()/(60*self.time_step))

        # Store everything in the object
        self.len_ones = duration_charging
        self.start_pos = start_pos


    def generate_contiguous_list(self,
                                 random: bool = True):
        """
        Generates a list with contiguous blocks of 1s and 0s where the series of 1s starts at a random position and has a random length.
        
        Parameters:
        random (bool): Have random time of stay or not.
        
        Returns:
        list: The generated list.
        """
        length = self.T
        if length <= 1:
            raise ValueError("Length of the list must be greater than 1.")
        
        if random : 
            # Determine the length of the series of 1s
            self.len_ones = random.randint(1, length - 1)
            # Determine the starting position for the series of 1s
            self.start_pos = random.randint(0, length - self.len_ones)
        
        else : 
            print('Not random')

        
        # Generate the list with contiguous blocks of 1s and 0s
        ls = [0] * length
        for i in range(self.len_ones):
            ls[self.start_pos + i] = 1
        return ls

    def generate_contiguous_matrix(self,
                                   SOC_random:bool = True,
                                   SOC_ini = None,
                                   SOC_final = None):
        """
        Generates a matrix with contiguous blocks of 1s and 0s in each column.
        
        Parameters:
        SOC_random (bool): If random the initial random 
        SOC_ini (np.array): Initial state of charge
        SOC_final (np.array) : desired final state of charge
        
        """
        rows = self.T
        cols = self.n_EV
        matrix = np.zeros((rows, cols), dtype=int)
        
        # Create random SOC and Schedule
        if SOC_random:
            self.SOC_ini = np.random.uniform(low=0.0, high=1.0, size= (self.n_EV,))
            self.SOC_final = np.ones((self.n_EV,))

            for col in range(cols):
                matrix[:, col] = self.generate_contiguous_list()

        # Create SOC and schedule based on inputs
        else : 
            self.SOC_ini =  SOC_ini
            self.SOC_final =  SOC_final

            for col in range(cols):
                matrix[:, col] = self.generate_contiguous_list(random=False)


        self.schedule_matrix = matrix



    def battery_dispatch(self,
                         charging_type:str = 'VPP'):
        """
        Optimize the battery dispatch using the price collected
        from RTE

        Args:
        charging_type (str) : smart, VPP, fast, aFRR biding 
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
        if charging_type == 'VPP':
            model.setObjective(quicksum(quicksum((x_ch[i,k] - x_dch[i,k]*self.eta)*self.price_ls[i]*self.delta_T*10**-3
                                                for i in range(T)) 
                                                for k in range(self.n_EV)), GRB.MINIMIZE)
        elif charging_type == 'smart':
            model.setObjective(quicksum(quicksum((x_ch[i,k])*self.price_ls[i]*self.delta_T*10**-3
                                    for i in range(T)) 
                                    for k in range(self.n_EV)), GRB.MINIMIZE)

        elif charging_type == 'fast':
            model.setObjective(quicksum(quicksum(((x_soc[i,k]))
                                    for i in range(T)) 
                                    for k in range(self.n_EV)), GRB.MAXIMIZE)
            
        elif charging_type == 'aFRR':
            vol_bids_up = model.addVars(T, lb=0.0,vtype=GRB.CONTINUOUS, name="bid_up") # bids up on aFFR in volume
            vol_bids_down = model.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="bid_down") # bids down on aFFR in volume
            bid_up = model.addVars(T, vtype=GRB.BINARY, name="bid_up") # bids up on aFFR in volume
            bid_down = model.addVars(T, vtype=GRB.BINARY, name="bid_down") # bids up on aFFR in volume

            # Set objective
            model.setObjective(quicksum(quicksum((- x_ch[i,k] + x_dch[i,k]*self.eta)*self.price_ls[i]*self.delta_T*10**-3
                                    for i in range(T)) 
                                    for k in range(self.n_EV)) +
                                    quicksum(bid_up[i]*vol_bids_up[i]*self.price_ls_aFFR_up[i] + 
                                             bid_down[i]*vol_bids_down[i]*self.price_ls_aFFR_down[i] for i in range(T)) , GRB.MAXIMIZE)
            
            # Constraints for aFFR bid
            for t in range(T):
                # (1) Cannot bid up and down at the same time
                model.addConstr(bid_up[t] + bid_down[t] <2 )

                # (2) Cannot bid more than all available energy
                model.addConstr( vol_bids_up[t] <= bid_up[t] * 
                                quicksum( x_soc[t,k] *self.capacity for k in range(self.n_EV)))
                
                # (3) Cannot bid down more than what EV can handle
                model.addConstr( vol_bids_down[t] <= bid_down[t] * 
                                quicksum( (1- x_soc[t,k]) *self.capacity for k in range(self.n_EV)))
                
                # (4) bid needs to be dispatch
                model.addConstr( vol_bids_down[t]*bid_down[t] <= quicksum(x_ch[t,k] for k in range(self.n_EV)))
                model.addConstr( vol_bids_up[t]*bid_up[t] <= quicksum(x_dch[t,k] for k in range(self.n_EV)))



        # Constraints (1)
        # Cannot discharge more than the available energy
        for t in range(T):
            for k in range(self.n_EV):
                model.addConstr(x_dch[t,k] <= x_dispatch[t,k]*x_soc[t,k]*self.capacity * self.schedule_matrix[t,k]/self.delta_T)
                model.addConstr(x_dch[t,k] <= x_dispatch[t,k]*self.P_dischar * self.schedule_matrix[t,k])
                model.addConstr(x_dch[t,k] <= self.P_charg_point)
        

        # Constraints (2)
        # Cannot charge more than available and cannot charge while dispatching
        for t in range(T): 
            for k in range(self.n_EV):
                model.addConstr(x_ch[t,k] <= (1-x_dispatch[t,k])*(1-x_soc[t,k])*self.capacity * self.schedule_matrix[t,k]/self.delta_T)
                model.addConstr(x_ch[t,k] <= (1-x_dispatch[t,k])*self.P_charg * self.schedule_matrix[t,k])
                model.addConstr(x_ch[t,k] <= self.P_charg_point)



        # Constraint (4)
        # Create the state of charge variable (start with full)
        # Start with a charged battery
        for k in range(self.n_EV):
            model.addConstr(x_soc[0,k] == self.SOC_ini[k])
            if np.sum(self.schedule_matrix[:,k])>5:
                model.addConstr(x_soc[T-1,k] >= self.SOC_final[k])
        
        # State of charge at each time step
        for t in range(1, T):
            for k in range(self.n_EV):
                model.addConstr((x_soc[t,k] - x_soc[t-1,k])*self.capacity == (x_ch[t-1,k] - x_dch[t-1,k])*self.delta_T)


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

        if charging_type in ['VPP', 'smart']:
            obj = model.getObjective()
            self.obj_value = obj.getValue()
            print(f'Objective value for {charging_type} = {self.obj_value}')
        else : 
            self.obj_value = 0
            for k in range(self.n_EV):
                self.obj_value += np.sum(charge[:,k]*self.price_ls[:]*self.delta_T*10**-3)
            print(f'Objective value for {charging_type} = {self.obj_value}')

        self.SOC = SOC
        return charge, discharge, SOC
    


    def generate_result_df(self):
        """
        Method that combine all the results in a dataframe
        The dataframe can be used for data visualization purpose
        """

        start_price_date = datetime(2024, 6, 19, 18, 00)
        end_price_date = datetime(2024, 6, 20, 17, 50)

        # Create the list of data
        date_list = []
        current_date = start_price_date

        while current_date <= end_price_date:
            date_list.append(current_date)
            current_date += timedelta(minutes=10)


        # Create the initial df with the SOC

        columns_name = [f'SOC {i}' for i in range(self.n_EV)]
        df = pd.DataFrame(self.SOC, columns=columns_name)

        # Add the price and the time
        df['Price [$/MWh]'] = self.price_ls
        df['Time'] = date_list

        return df