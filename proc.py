import os
import numpy as np
import utils
import model as m 
from save import save_one_neuron_exp

class Process():
    def __init__(self, V_th, E_L, dt, Lt, range_t, N, nu, time_step, Iinj, alfa, num_inputs, input, neuron) -> None:
        self.V_th = V_th
        self.E_L = E_L
        self.dt = dt
        self.Lt = Lt
        self.nu = nu
        self.N = N
        self.time_step = time_step
        self.range_t = range_t
        self.Iinj = Iinj
        self.alfa = alfa
        self.in_features = num_inputs
        self.input = input
        self.neuron = neuron
        self.weights= np.random.rand(num_inputs, 1)
        #self.weights = np.ones((in_features, out_features))

    def plot_files(self, folder_name):
            lst = os.listdir(folder_name)
            lst = [int(file.split('_')[-1][:-4]) for file in lst]
            indxes = np.argsort(lst)
            file_names =  os.listdir(folder_name)
            for i, indx in enumerate(indxes):
                    file_name = os.path.join(folder_name,file_names[indx])
                    V, out_spikes, in_spikes, _, prop, nu, weights = utils.get_csv_file_data(file_name)
                    print(weights)
                    time, time_step, N, alfa, in_features = prop
                    Iinj = np.max(in_spikes)
                    range_t = np.arange(0, time, 0.1)
                    utils.plot_spikes(int(in_features), 1, in_spikes, out_spikes.T, V, range_t, self.V_th, self.E_L, 
                                      legend = [nu, f"{time_step}", f"{N}", f"{alfa}", f"{Iinj}"])

    def plot_file(self, file_name, acc:bool):
            V, out_spikes, in_spikes, _, prop, nu, weights = utils.get_csv_file_data(file_name)
            print(weights)
            time, time_step, N, alfa, in_features = prop
            range_t = np.arange(0, time, 0.1)
            Iinj = np.max(in_spikes)
            utils.plot_spikes(int(in_features), 1, in_spikes, out_spikes.T, V, range_t, self.V_th, self.E_L, 
                              legend = [nu, f"{time_step}", f"{N}", f"{alfa}",  f"{Iinj}"])
            if acc:
                    accuracy = utils.count_acc(in_spikes[0], out_spikes[:, 0], time_step/self.dt)
                    print(accuracy)
                    return accuracy
            return None

    def plot_folder(self, folder_name, acc:bool):
            V, out_spikes, in_spikes, _, prop, nu, weights = utils.get_csv_dir_data(folder_name)
            print(weights)
            time, time_step, N, alfa, in_features = prop
            range_t = np.arange(0, time, 0.1)
            Iinj = np.max(in_spikes)
            utils.plot_spikes(int(in_features), 1, in_spikes, out_spikes.T, V, range_t, self.V_th, self.E_L, 
                              legend = [nu, f"{time_step}", f"{N}", f"{alfa}", f"{Iinj}"])
            if acc:
                    accuracy = utils.count_acc(in_spikes[0], out_spikes[:, 0], time_step/self.dt)
                    print(accuracy)
                    return accuracy
            return None

    def make_new_data(self, epochs, train):
            print(self.weights)
            if not os.path.isdir("output_data"):
                    os.mkdir('output_data')
            start_V = self.E_L
            i = 0 
            layer1 = m.Layer(self.in_features, 1, self.weights, start_V, self.neuron)
            #np.random.seed(1)
            snn = m.FC([layer1], self.input, self.Lt, 1, N_spk=self.N, nu=self.nu, time_step=self.time_step, train=train, dVs=0, 
                        check=True, period=i)
            in_spikes, out_spikes, V, dV, weights = snn.forward()
            save_one_neuron_exp(file_name=f'output_data/data_{i}.csv', dV=dV, V=V, out_spikes=out_spikes, in_spikes=in_spikes, input=self.input,
                                spk_amp=self.neuron.spk_amp, L_time=self.Lt, dt = self.dt, t_step=self.time_step, N_spk=self.N, alfa=self.alfa, nu = self.nu, weights=weights)
            i+=1
            while i < epochs:
                    V, _, _, dV, _, _, weights = utils.get_csv_dir_data("output_data")
                    print(weights)
                    start_V = V[-1, 0]
                    layer1 = m.Layer(self.in_features, 1, weights, start_V, self.neuron)
                    #np.random.seed(1)
                    snn = m.FC([layer1], self.input, self.Lt, 1, N_spk=self.N, nu=self.nu, time_step=self.time_step, 
                            train=train, dVs=dV, check=True, period=i)
                    in_spikes, out_spikes, V, dV, weights = snn.forward()
                    save_one_neuron_exp(file_name=f'output_data/data_{i}.csv', dV=dV, V=V, out_spikes=out_spikes, in_spikes=in_spikes, input=self.input,
                                spk_amp=self.neuron.spk_amp, L_time=self.Lt, dt = self.dt, t_step=self.time_step, N_spk=self.N, alfa=self.alfa, nu = self.nu, weights=weights)
                    i+=1

    def add_data(self, epochs, i, train):
            while i < epochs:
                    V, _, _, dV, _, _, weights = utils.get_csv_dir_data("output_data")
                    start_V = V[-1, 0]
                    layer1 = m.Layer(self.in_features, 1, weights, start_V, self.neuron)
                    #np.random.seed(12)
                    snn = m.FC([layer1], self.input, self.Lt, 1, N_spk=self.N, nu=self.nu, time_step=self.time_step, 
                            train=train, dVs=dV, check=True, period = i)
                    in_spikes, out_spikes, V, dV, weights = snn.forward()
                    save_one_neuron_exp(file_name=f'output_data/data_{i}.csv', dV=dV, V=V, out_spikes=out_spikes, in_spikes=in_spikes, input=self.input,
                                spk_amp=self.neuron.spk_amp, L_time=self.Lt, dt = self.dt, t_step=self.time_step, N_spk=self.N, alfa=self.alfa, nu = self.nu, weights = weights)
                    i+=1

    def instant_view(self, train):
            #np.random.seed(12)
            start_V = self.E_L
            layer1 = m.Layer(self.in_features, 1, self.weights, start_V, self.neuron)
            snn = m.FC([layer1], self.input, self.Lt, 1, N_spk=self.N, nu=self.nu, time_step=self.time_step, train=train, 
                        dVs=0, check=False, period=0)
            in_spikes, out_spikes, V = snn.forward()
            print(f"number of input spikes: {in_spikes[in_spikes==1].shape}")
            utils.plot_spikes(self.in_features, 1, in_spikes*self.Iinj, out_spikes.T, V, self.range_t, self.V_th, self.E_L, 
                              legend = [self.nu, f"{self.time_step}", f"{self.N}", f"{self.alfa}", f"{self.Iinj}"])
            #accuracy = utils.count_acc(in_spikes[0], out_spikes[:, 0], self.time_step/self.dt)
            #print(accuracy)
