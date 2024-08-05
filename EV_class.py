import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta
from RTE_API import *

class EV_BATTERY(RTE_API):
    
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
        super().__init__()
        np.random.seed(0)
        self.P_charg = P_charg
        self.capacity = capacity
        self.P_dischar = P_dischar
        self.eta = eta
        self.n_EV = n_EV

    def define_path_and_inputs(self,
                           path_to_price: str,
                           path_to_aFFR: str,
                           path_parking_data: str,
                           charging_schedule: tuple,
                            SOC_ini: np.array,
                            SOC_final: np.array,
                            price_API: bool = False):
        """
        Define the path to the important csv files and the inputs from
        the web app

        Args : 

            charging_schedule : tuple of datetime object of format "MM/DD HH:mm"
        """
        self.path_to_price = path_to_price
        self.path_to_aFFR = path_to_aFFR
        self.path_parking_data = path_parking_data
        self.charging_schedule = charging_schedule
        self.SOC_ini_read = SOC_ini
        self.SOC_final_read =SOC_final
        self.price_API = price_API

    def read_spot_price(self, 
                        resample = False,
                        sampling = 6):
        """
        Read the df of price
        """
        if self.price_API :
            # Generate the RTE token to access the data
            self.generate_token()

            # Generate the data
            self.get_wholesale_data()

            # Generate the list of price
            self.price_ls = self.wholesale_data_df['price']

        else : 
            # Read the dataframe
            df_price_spot = pd.read_csv(self.path_to_price)
            # generate price list
            self.price_ls = df_price_spot['Prix (€/MWh)']

        self.delta_T =1
        self.time_step = 60
        if resample :
            self.delta_T = 1/sampling
            self.time_step = 60*self.delta_T
            resampled_list = []
            for elem in self.price_ls:
                resampled_list.extend([elem] * sampling)
            self.price_ls = resampled_list

        self.T = len(self.price_ls)

    def read_aFFR_price(self,
                        date_range : list):
        """
        Read the aFRR price data in the date range indicated 
        in the function 

        Args  :
        path_to_aFFR (str) : csv file from RTE with the data
        date_range (list) : list of string containing the date range to consider
        """

        df_aFRR_price = pd.read_csv(self.path_to_aFFR)
        
        # Split the column into two columns
        df_aFRR_price[['Start Time', 'End Time']] = df_aFRR_price['ISP CET/CEST'].str.split(' - ', expand=True)

        # Convert to datetime
        df_aFRR_price['Start Time'] = pd.to_datetime(df_aFRR_price['Start Time'], format='%d/%m/%Y %H:%M:%S')
        df_aFRR_price['End Time'] = pd.to_datetime(df_aFRR_price['End Time'], format='%d/%m/%Y %H:%M:%S')


        # Define the start and end datetime for the range
        start_date = pd.to_datetime(date_range[0], format='%d/%m/%Y %H:%M')
        end_date = pd.to_datetime(date_range[1], format='%d/%m/%Y %H:%M')


        # Filter the dataframe to keep rows within the date range
        filtered_df = df_aFRR_price[(df_aFRR_price['Start Time'] >= start_date) & (df_aFRR_price['Start Time'] <= end_date)]


        filtered_df['Price Up (EUR/MWh)'] = filtered_df['Price Up (EUR/MWh)'].replace(' ', '0')
        filtered_df['Price Up (EUR/MWh)'] = filtered_df['Price Up (EUR/MWh)'].replace('', '0')
        filtered_df['Price Down (EUR/MWh)'] = filtered_df['Price Down (EUR/MWh)'].replace(' ','0')
        filtered_df['Price Down (EUR/MWh)'] = filtered_df['Price Down (EUR/MWh)'].replace('','0')

        filtered_df['Price Up (EUR/MWh)'] = filtered_df['Price Up (EUR/MWh)'].astype(float)
        filtered_df['Price Down (EUR/MWh)'] = filtered_df['Price Down (EUR/MWh)'].astype(float)
        # Save the data in the list
        self.price_ls_aFFR_up = filtered_df['Price Up (EUR/MWh)'].values
        self.price_ls_aFFR_down = filtered_df['Price Down (EUR/MWh)'].values

        for elem in filtered_df['Price Up (EUR/MWh)'].values:
            print(elem)
        for elem_1 in self.price_ls_aFFR_up:
            print(elem_1)

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
        

    def reformate_charing_time(self):
        """
        Reformate the charging schedule to make it suitable with 
        the optimization process

        Args : 
        charging_schedule : tuple of datetime object of format "MM/DD HH:mm"
        """
        start_price_date = datetime(2024, 6, 19, 18, 00)
        #Determine the start position for the optimization
        detla_time_start = self.charging_schedule[0] - start_price_date
        # for now the steps are of 10 minutes
        start_pos = int(detla_time_start.total_seconds()/(60*self.time_step))


        # Determine the charging duration 
        detla_time_charge = self.charging_schedule[1] -  self.charging_schedule[0]
        duration_charging = int(detla_time_charge.total_seconds()/(60*self.time_step))

        # Store everything in the object
        self.len_ones = duration_charging
        self.start_pos = start_pos

    def robust_parked_time(self,
                           confidence_level:int = 2):
        """
        Method that derive the statistic of the parked time,
        used to define the uncertainties set of the robust optimization 
        method

        Args:
            path_parking_data (str) : path to the NREL data
        """

        df_parked_time = pd.read_csv(self.path_parking_data)

        #Mean and std in hours of time parked
        mu = np.mean(df_parked_time['Rolling_diff'])
        sigma = np.std(df_parked_time['Rolling_diff'])

        worst_duration_charging = int((mu - confidence_level*sigma)*3600/(60*self.time_step))

        # Store everything in the object
        self.len_ones = worst_duration_charging
 
    def build_charging_schedule(self,
                                schedule_type: str):
        """
        Method that build the charging schedule based on different option

        Args : 
            schedule_type (str) : need to be 'deterministic', 'robust+known start',
                                'robust + fully random', 'fully random'
        """
        rows = self.T
        cols = self.n_EV
        matrix = np.zeros((rows, cols), dtype=int)

        # generate the matrix based on the schedule type
        for col in range(cols):
                matrix[:, col] = self.generate_contiguous_list(schedule_type)

        self.schedule_matrix = matrix


    def generate_contiguous_list(self,
                                 schedule_type: str):
        """
        Generates a list with contiguous blocks of 1s and 0s where the series
        of 1s starts at a random position and has a random length.
        
        Parameters:
        schedule_type (str): type of charging schedule needs to be 'deterministic',
                            'robust+known start', 'robsut+random', 'fully random'
        
        Returns:
        list: The generated list.
        """
        length = self.T
        if length <= 1:
            raise ValueError("Length of the list must be greater than 1.")
        

        if schedule_type == 'deterministic':
            #If not random => call the function reformate_charging_time 
            # to create the contiguous list with the random length
            self.reformate_charing_time()

        elif schedule_type == 'robust+known start':
            # Read the charging data
            self.reformate_charing_time()
            # Create the robust layer
            self.robust_parked_time()

        elif schedule_type == 'robsut+random':
            # Create the robust layer
            self.robust_parked_time()
            # Determine the starting position for the series of 1s
            self.start_pos = random.randint(0, length - self.len_ones)

        elif schedule_type == 'fully random':
            # Determine the length of the series of 1s
            self.len_ones = random.randint(1, length - 1)
            # Determine the starting position for the series of 1s
            self.start_pos = random.randint(0, length - self.len_ones)

        # Generate the list with contiguous blocks of 1s and 0s
        ls = [0] * length
        for i in range(self.len_ones):
            ls[self.start_pos + i] = 1
        return ls

    def create_SOC(self,
                   SOC_type : str):
        """
        Method to read the SOC 
        """

        if SOC_type == 'deterministic':
            # Read directly from the dashboard
            self.SOC_ini =  self.SOC_ini_read
            self.SOC_final =  self.SOC_final_read
        
        elif SOC_type == 'random':
            self.SOC_ini = np.random.uniform(low=0.0, high=1.0, size= (self.n_EV,))
            self.SOC_final = np.ones((self.n_EV,))

    def generate_contiguous_matrix(self,
                                   SOC_random:bool = True,
                                   SOC_ini = None,
                                   SOC_final = None,
                                   data_driven = False,
                                   random_schedule = False):
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
                matrix[:, col] = self.generate_contiguous_list(random_schedule=random_schedule, data_driven=data_driven)


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
        model.setParam("TimeLimit", 5) 


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
            bid_up = model.addVars(T, vtype=GRB.BINARY, name="bid_up") # decision to bid up
            bid_down = model.addVars(T, vtype=GRB.BINARY, name="bid_down") # decision to bid down

            # Set objective
            model.setObjective(quicksum(quicksum((- x_ch[i,k] + x_dch[i,k]*self.eta)*self.price_ls[i]*self.delta_T*10**-3
                                    for i in range(T)) 
                                    for k in range(self.n_EV)) +
                                    quicksum((bid_up[i]*vol_bids_up[i]*self.price_ls_aFFR_up[i] + 
                                             bid_down[i]*vol_bids_down[i]*self.price_ls_aFFR_down[i])*self.delta_T*10**-3 for i in range(T)) , GRB.MAXIMIZE)
            
            # Constraints for aFFR bid
            for t in range(T):
                # (1) Cannot bid up and down at the same time
                model.addConstr(bid_up[t] + bid_down[t] <= 1 )

                # (2) Cannot bid more than all available energy
                model.addConstr( vol_bids_up[t] <= bid_up[t] * 
                                quicksum( x_soc[t,k] *self.capacity for k in range(self.n_EV)))
                
                # (3) Cannot bid down more than what EV can handle
                model.addConstr( vol_bids_down[t] <= bid_down[t] * 
                                quicksum( (1- x_soc[t,k]) *self.capacity for k in range(self.n_EV)))
                
                # (4) bid needs to be dispatch
                model.addConstr( vol_bids_down[t]*bid_down[t] <= quicksum(x_ch[t,k] for k in range(self.n_EV)))
                model.addConstr( vol_bids_up[t]*bid_up[t] <= quicksum(x_dch[t,k] for k in range(self.n_EV)))

                # (5) need to bid at least 1MW up and down
                model.addConstr( vol_bids_down[t] >= 1*bid_down[t])
                model.addConstr( vol_bids_up[t] >= 1*bid_up[t])




        # Constraints (1)
        # Cannot discharge more than the available energy
        for t in range(T):
            for k in range(self.n_EV):
                model.addConstr(x_dch[t,k] <= x_soc[t,k]*self.capacity * self.schedule_matrix[t,k]/self.delta_T)
                model.addConstr(x_dch[t,k] <= x_dispatch[t,k]*self.P_dischar * self.schedule_matrix[t,k])
                model.addConstr(x_dch[t,k] <= self.P_charg_point)
        

        # Constraints (2)
        # Cannot charge more than available and cannot charge while dispatching
        for t in range(T): 
            for k in range(self.n_EV):
                model.addConstr(x_ch[t,k] <= (1-x_soc[t,k])*self.capacity * self.schedule_matrix[t,k]/self.delta_T)
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

        # Charge state
        charge = np.zeros((T,self.n_EV))
        
        #Power of each EV battery
        discharge = np.zeros((T,self.n_EV))
        SOC = np.zeros((T,self.n_EV))

        # Save the variables
        for j in range(T):
            for k in range(self.n_EV):
                charge[j,k] = x_ch[j,k].X
                discharge[j,k] = x_dch[j,k].X
                SOC[j,k] = x_soc[j,k].X

        if charging_type in ['VPP', 'smart', 'aFRR']:
            obj = model.getObjective()
            self.obj_value = obj.getValue()
            print(f'Objective value for {charging_type} = {self.obj_value}')
        else : 
            self.obj_value = 0
            for k in range(self.n_EV):
                self.obj_value += np.sum(charge[:,k]*self.price_ls[:]*self.delta_T*10**-3)
            print(f'Objective value for {charging_type} = {self.obj_value}')

        if charging_type == 'aFRR':
            self.bid_up_ls = np.zeros((T,))
            self.bid_down_ls = np.zeros((T,))
            for t in range(T):
                self.bid_up_ls[t] = vol_bids_up[t].X
                self.bid_down_ls[t] = vol_bids_down[t].X

        self.SOC = SOC
        return charge, discharge, SOC
    


    def generate_result_df(self):
        """
        Method that combine all the results in a dataframe
        The dataframe can be used for data visualization purpose
        """

        if self.price_API:
            start_price_date = pd.to_datetime(self.wholesale_data_df['start_date'].iloc[0])
            end_price_date = pd.to_datetime(self.wholesale_data_df['end_date'].iloc[-1]) - timedelta(minutes=10)
            print(f'End Date ={end_price_date}')
        else:
            start_price_date = datetime(2024, 6, 19, 18, 00)
            end_price_date = datetime(2024, 6, 20, 17, 50)

        # Create the list of data
        date_list = []
        current_date = start_price_date

        while current_date <= end_price_date:
            date_list.append(current_date)
            current_date += timedelta(minutes=15)


        # Create the initial df with the SOC

        columns_name = [f'SOC {i}' for i in range(self.n_EV)]
        df = pd.DataFrame(self.SOC, columns=columns_name)

        # Add the price and the time
        df['Price [$/MWh]'] = self.price_ls
        df['Time'] = date_list

        return df