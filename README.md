<p align="center">
  <img src = "https://github.com/Wormsin/fractional-snn/assets/142012648/cc7e4c98-3417-468f-b96e-7ed672fb632f" width="442" height="252">
  <img src = "https://github.com/Wormsin/fractional-snn/assets/142012648/24d818ff-feb3-4ad1-b506-401ed0f206cd" width="442" height="252">
</p>

# Fractional order Leaky Integrate-and-Fire Neuron Model
This project implements processing of intermittent spike trains by fractional leaky integrate-and-fire neural model. \
Thus the input data is encoded with a fractional Poisson distribution with a long-term memory parameter **nu**.
## model.py 
This code implements a fully connected neural network. You can configure an arbitrary number of layers. The fractional differential equation is simplified according to the L1 scheme. 
## main.py
**proc.py** is made for experiments on _one_ neuron with several inputs. \

Create and save new data in the folder "output_data". \
**epochs** are the number of experiments with a given observation time that are dependent on each other.
So each epoch, **voltage-memory-trace** is considered for the entire observation time, i.e. it consists of all files.
```ruby
proc.make_new_data(epochs)
```
Create and add new data to the existing "output_data". These experiments may have different observation times and neuron configurations.
```ruby
proc.add_data(final_index, start_index)
```
**proc.plot_file(...)** -- plot data from one csv file, \
**proc.plot_folder(...)** -- plot data from a folder as one expirement, i.e. concatenating voltage, **voltage-memory-trace** and spikes values, \
**proc.plot_files(...)** -- plot files one after another.
```ruby
proc.plot_file(file_name = "output_data/data_4.csv", acc = False)
proc.plot_folder(folder_name = "output_data", acc = False)
proc.plot_files(folder_name = "output_data")
```
**utils.py** has functions for calculating and plotting ISI time values. _(interspike intervals time values)_ \
**save.py** has function for saving data into csv file including **voltage-memory-trace**. 
## model parameters
| Parameters | Values | Description                                                          |
|------------|--------|----------------------------------------------------------------------|
| Cm         | 0.5 nF | membrane capacity                                                    |
| Vreset     | -70 mV | membrane potential after reset                                       |
| Vth        | -50 mV | threshold voltage                                                    |
| V0         | -70 mV | initial membrane potential                                           |
| gL         | 25 nS  | leakage conductivity                                                 |
| tref       | 5 ms   | recovery period                                                      |
| stdp_rate  | 0.0625 | learning rate for stdp                                               |
| tm         | 5 ms   | time constant for negative  stdp function branch                     |
| tp         | 3 ms   | time constant for negative  stdp function branch                     |
| Am         | -0.3   | amplitude constant for negative stdp function branch                 |
| Ap         | 0.6    | amplitude constant for negative stdp function branch                 |
| dt         | 0.1 ms | time per step                                                        |
| range_t    | x ms   | observation time                                                     |
| time_step  | x, ms  | duration of a step impulse                                           |
| N          | x      | number of pulses per range_t                                         |
| nu         | [x]    | long-term memory parameters of the input  fractional Poisson process |
| alpha      | x      | parameter of LIF neuron                                              |
| input      | [x]    | coefficients for input spikes rate                                   |


