import os
import numpy as np
import h5py
import scipy.interpolate

import statsmodels.api as sm

import matplotlib.pyplot as plt

PERTURBATION_DELAY = 0.002

class triangleData(object):
    def __init__(self, filename):
        self.filename = filename

        _, ext = os.path.splitext(filename)

        if (ext.lower() == '.h5') or (ext.lower() == '.hdf5'):
            self.loadH5data()
        else:
            raise ValueError('Unrecognized file type {}'.format(filename))

        self.processActivation()
        self.processTriangles()

    def loadH5data(self):
        with h5py.File(self.filename, 'r') as file:
            self.length = np.array(file['/RawInput/Length'])
            self.force = np.array(file['/RawInput/Force'])
            self.stim = np.array(file['/RawInput/Stim voltage'])

            self.wait_before = file['/NominalStimulus'].attrs['WaitPre']
            self.wait_after = file['/NominalStimulus'].attrs['WaitPost']
            self.amplitude = file['/NominalStimulus'].attrs['Amplitude']
            self.frequency = file['/NominalStimulus'].attrs['Frequency']
            self.ncycles = file['/NominalStimulus'].attrs['Cycles']

            self.isactive = file['/Output'].attrs['ActivationOn']
            self.activation_start_cycle = file['/Output'].attrs['ActivationStartCycle']
            self.activation_start_phase = file['/Output'].attrs['ActivationStartPhase'] / 100.0
            self.activation_duty = file['/Output'].attrs['ActivationDuty'] / 100.0

            self.ispert = file['/ParameterTree/Stimulus/Perturbations'].attrs['Type'] == b'Triangles'
            if self.ispert:
                self.perturbation_amplitude = file['/ParameterTree/Stimulus/Perturbations/Parameters'].attrs['Amplitude']
                self.perturbation_duration = file['/ParameterTree/Stimulus/Perturbations/Parameters'].attrs['Duration']
                self.perturbation_phase = file['/ParameterTree/Stimulus/Perturbations/Parameters'].attrs['Phase']
                self.perturbation_delay = file['/ParameterTree/Stimulus/Perturbations/Parameters'].attrs['Delay in between']
                self.perturbation_start_cycle = file['/ParameterTree/Stimulus/Perturbations/Parameters'].attrs['Start cycle']
                self.nperturbations = file['/ParameterTree/Stimulus/Perturbations/Parameters'].attrs['Repetitions']
            else:
                self.perturbation_amplitude = 0.0
                self.perturbation_duration = 0.0
                self.perturbation_phase = None
                self.perturbation_start_cycle = 0
                self.nperturbations = 0

            self.t = np.array(file['/NominalStimulus/t'])

            self.length_scale = file['/ParameterTree/Motor parameters'].attrs['Length scale']
            self.force_scale = file['/ParameterTree/DAQ/Input'].attrs['Force scale']

            self.length *= self.length_scale
            self.force *= self.force_scale

    def processActivation(self):
        t_active = []
        for c in range(self.activation_start_cycle, self.ncycles):
            ton = (c + 0.25 + self.activation_start_phase) / self.frequency
            toff = ton + self.activation_duty / self.frequency

            t_active.append([ton, toff])

        self.t_active = np.array(t_active)

    def processTriangles(self):
        tpert = (self.perturbation_start_cycle + self.perturbation_phase) / self.frequency + \
            np.arange(self.nperturbations-1)*self.perturbation_delay + PERTURBATION_DELAY
        ton = tpert - self.perturbation_duration / 2
        toff = tpert + self.perturbation_duration / 2

        self.t_pert_onoff = np.vstack((ton, toff)).T
        self.t_pert = tpert

    def extractTriangles(self, tpre=0.1, tpost=0.3, smooth=None):
        dt = self.t[1] - self.t[0]

        std = np.std(self.length[self.t < -0.01])

        tpert = []
        length_pert = []
        vel_pert = []
        force_pert = []

        for p1, pctr1 in zip(self.t_pert_onoff, self.t_pert):
            ispert = np.logical_and(self.t >= p1[0] - tpre, self.t < p1[1] + tpost)

            t1 = self.t[ispert] - pctr1
            tpert.append(t1)

            l1 = self.length[ispert]

            if smooth is None:
                length_pert.append(l1)
                l1s = l1
            else:
                w1 = np.full_like(l1, 1.0/std)

                sp = scipy.interpolate.UnivariateSpline(t1, l1, w=w1, s=smooth*len(l1))
                l1s = sp(t1)

                length_pert.append(l1s)

            vel_pert.append(np.gradient(l1s, dt))
            force_pert.append(self.force[ispert])

        return np.array(tpert), np.array(length_pert), np.array(vel_pert), np.array(force_pert)

    def fit_stiffness_damping(self):
        tpert, length_pert, vel_pert, force_pert = self.extractTriangles(tpre=0, tpost=0, smooth=0.95)

        for f1 in force_pert:
            f1 -= np.mean(f1[:10])

        X = np.vstack((np.ones_like(length_pert.flat), length_pert.flat, vel_pert.flat)).T
        model = sm.OLS(force_pert.flatten(), X)
        fit = model.fit()

        print(fit.summary())
        force_pred = fit.predict()
        force_pred = np.reshape(force_pred, tpert.shape)

        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].plot(tpert, force_pert, 'r-')
        ax[0].plot(tpert, force_pred, 'b--')

        ax[1].plot(tpert, length_pert, 'b-')
        ax[2].plot(tpert, vel_pert, 'g-')

    def plot(self):
        fig, ax = plt.subplots(3, 1, sharex=True)

        ax[0].plot(self.t, self.length)
        ax[0].set_ylabel('Length')

        ax[1].plot(self.t, self.force)
        ax[1].set_ylabel('Force')

        ax[2].plot(self.t, self.stim)
        ax[2].set_ylabel('Stim')

        for act1 in self.t_active:
            for ax1 in ax:
                ax1.axvspan(xmin=act1[0], xmax=act1[1], facecolor='k', alpha=0.3)

        for p1 in self.t_pert_onoff:
            for ax1 in ax:
                ax1.axvspan(xmin=p1[0], xmax=p1[1], facecolor='r', alpha=0.3)

    def plot_perturbations(self):
        tpert, length_pert, vel_pert, force_pert = self.extractTriangles(smooth=0.95)

        fig, ax = plt.subplots(3, 1, sharex=True)

        for t1, l1, v1, f1 in zip(tpert, length_pert, vel_pert, force_pert):
            ax[0].plot(t1, l1, 'k-')
            ax[1].plot(t1, v1, 'b-')
            ax[2].plot(t1, f1, 'r-')

def main():
    filename = './rawdata/20180629_L131/MUSCLE A/006-stim-15-25pert-act.h5'

    tridata = triangleData(filename)

    plt.ion()

    tridata.plot()
    tridata.plot_perturbations()

    tridata.fit_stiffness_damping()

    plt.show(block=True)

main()
