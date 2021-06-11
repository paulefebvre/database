#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 13:28:04 2021

@author: cghiaus
Cooling as a control & parameter optilizaton problem OOP
Cooling & desumidification, reheating

GENERALITIES
========================================================================
Units
Temperatures: °C
Humidity ration: kg_vapor / kg_dry_air
Relative humidity: -
Heat flow rate: W
Mass flow rate: kg/s

Points on psychrometric chart (θ, w):
o) out      outdoor
0) s        saturation
1) S        supply air
2) I        indoor


System as a direct problem
--------------------------
 out        s              S          I
==o==>[CC]==0==>[HC]==F==1===>[TZ]==2==>
       /\\      /    /      / /
      t  sl    s    m      s l

Inputs:
o       outdoor temperature & humidity ratio
QtCC        total heat load of CC
QsHC        sensible heat load of HC
QsTZ        sensible heat load of TZ
QlTZ        latent heat load of TZ
Parameter:
m       mass flow rate of dry air

Elements (8 equations):
CC          cooling coil (4 equations)
HC          heating coil (2 equations)
TZ          thermal zone (2 equations)
F           fan (m is a given parameter)

Outputs (8 unknowns):
0, 1, 2     temperature and humidity ratio (6 unknowns)
QsCC, QlCC  sensible and latent heat of CC (2 unknowns)

CAV System with linear controllers for θI & φI
----------------------------------------------
 out        s              S          I
==0==>[CC]==1==>[HC]===F===2===>[TZ]==3==>
       /\\      /     /         //    ||
      t  sl    s     m         sl     ||
      |        |                      ||
      |        |                      ||
      |        |<------[K]------------||<-wI<-φI
      |<---------------[K]------------|<-θI

Inputs:
θo, φo      outdoor temperature & relative humidity
θI, φI      indoor air temperature and humidity
QsTZ        sensible heat load of TZ
QlTZ        latent heat load of TZ
Parameter:
m           mass flow rate of dry air

Elements (10 equations):
CC          cooling coil (4 equations)
HC          heating coil (2 equations)
TZ          thermal zone (2 equations)
F           fan (no equation, m is given)
KθI         indoor temperature controller (1 equation)
KwI         indoor humidity controller (1 equation)

Outputs (10 unknowns):
0, 1, 2     temperature and humidity ratio (6 unknowns)
QsCC, QlCC  sensible and latent heat of CC (2 unknowns)
QtCC        total heat load of CC (1 unknown)
QsHC        sensible heat load of HC (1 unknown)

VAV System with linear & least-squares controllers for θI, φI & θS
------------------------------------------------------------------
 out        s              S          I
==0==>[CC]==1==>[HC]===F===2===>[TZ]==3==>
       /\\      /     /    |    //    ||
      t  sl    s     m     |   sl     ||
      |        |     |     |          ||
      |        |     |<-ls-|<-θS      ||
      |        |<------[K]------------|<-wI
      |<---------------[K]------------|<-θI

Inputs:
θo, φo      outdoor temperature & relative humidity
θI, φI      indoor air temperature and humidity
θS          supply air temperature
QsTZ        sensible heat load of TZ
QlTZ        latent heat load of TZ

Elements (11 equations):
CC          cooling coil (4 equations)
HC          heating coil (2 equations)
TZ          thermal zone (2 equations)
F           fan (m is given)
KθI         indoor temperature controller (1 equation)
KwI         indoor humidity controller (1 equation)
lsθS        mass flow rate of dry air controller (1 non-linear equation)

Outputs (11 unknowns):
0, 1, 2     temperature and humidity ratio (6 unknowns)
QsCC, QlCC  sensible and latent heat of CC (2 unknowns)
QtCC        total heat load of CC (1 unknown)
QsHC        sensible heat load of HC (1 unknown)
Parameter:
m           mass flow rate of dry air (1 unknown)


CONTENTS (methods)
========================================================================
__init__    Initialization of CcTZ object.
lin_model   Solves the set of linear equations
            with saturation curve linearized around ts0
solve_lin   Solves iteratively the lin_model s.t. the error of
            humid. ratio between two iterrations is approx. zero
            (i.e. solves ws = f(θs) for saturation curve).
solve_ls    Finds m s.t. θS = θSsp (solves θS - θSsp = 0 for m).
            Uses least-squares to find m that minimizes θS - θSsp
psy_chart   Draws psychrometric chart (imported from psychro)
CAV_wd      CAV to be used in Jupyter widgets.
            solve_lin and draws psy_chart.
VAV_wd      VAV to be used in Jupyter widgets.
            solve_ls and draws psy_chart.
"""
import numpy as np
import pandas as pd
import psychro as psy

# constants
c = 1e3         # J/kg K, air specific heat
l = 2496e3      # J/kg, latent heat

# to be used in self.solve_ls / least_squares
m_max = 100     # ks/s, max dry air mass flow rate
θs_0 = 5        # °C, initial guess for saturation temperature


class CcTz:
    """
    Cooling coil and thermal zone
    """

    def __init__(self, m, θo, φo, tIsp, φIsp, QsTZ, QlTZ):
        self.design = np.array([m, θo, φo, tIsp, φIsp, QsTZ, QlTZ])
        self.actual = np.array([m, θo, φo, tIsp, φIsp, QsTZ, QlTZ])

    def lin_model(self, θs0):
        """
        Linearized model. Solves a set of 10 linear equations.
        Saturation curve is linearized in ts0. The point (θs, ws) is not on
        the saturation curve (ws = f(θs) is not satisfied).

        Parameters
        ----------
        θs0     °C, initial guess of saturation temperature

        Parameters from object
        ---------------------
        m, θo, φo, tIsp, φIsp, QsTZ, QlTZ = self.actual

        Returns (10 unknowns)
        ---------------------
        x : θs, ws, θS, wS, θI, wI, QsCC, QlCC, QtCC, QsHC

         out        s              S          I
        ==0==>[CC]==1==>[HC]===F===2===>[TZ]==3==>
               /\\      /     /         //    ||
              t  sl    s     m         sl     ||
              |        |                      ||
              |        |                      ||
              |        |<------[K]------------||<-wI<-φI
              |<---------------[K]------------|<-θI

        Inputs
        θs0         saturation temperature used for linearization
        from self.actual:
        θo, φo      outdoor temperature & relative humidity
        θI, φI      indoor air temperature and humidity
        QsTZ        sensible heat load of TZ
        QlTZ        latent heat load of TZ
        m           mass flow rate of dry air

        Elements (10 equations):
        CC          cooling coil (4 equations)
        HC          heating coil (2 equations)
        TZ          thermal zone (2 equations)
        F           fan (no equation, m is given)
        KθI         indoor temperature controller (1 equation)
        KwI         indoor humidity controller (1 equation)

        Outputs (10 unknowns):
        0, 1, 2     temperature and humidity ratio (6 unknowns)
        QsCC, QlCC  sensible and latent heat of CC (2 unknowns)
        QtCC        total heat load of CC (1 unknown)
        QsHC        sensible heat load of HC (1 unknown)
        """
        m, θo, φo, θIsp, φIsp, QsTZ, QlTZ = self.actual
        Kθ, Kw = 1e10, 1e10             # controller gain
        wO = psy.w(θo, φo)            # hum. out

        A = np.zeros((10, 10))          # coefficents of unknowns
        b = np.zeros(10)                # vector of inputs
        # CC cooling coil
        A[0, 0], A[0, 6], b[0] = m * c, -1, m * c * θo
        A[1, 1], A[1, 7], b[1] = m * l, -1, m * l * wO
        A[2, 0], A[2, 1] = psy.wsp(θs0), -1
        b[2] = psy.wsp(θs0) * θs0 - psy.w(θs0, 1)
        A[3, 6], A[3, 7], A[3, 8], b[3] = 1, 1, -1, 0
        # HC heating coil
        A[4, 0], A[4, 2], A[4, 9], b[4] = m * c, -m * c, 1, 0
        A[5, 1], A[5, 3], b[5] = m * l, -m * l, 0
        # TZ thermal zone
        A[6, 2], A[6, 4], b[6] = m * c, -m * c, -QsTZ
        A[7, 3], A[7, 5], b[7] = m * l, -m * l, -QlTZ
        # Kt indoor temperature controller
        A[8, 4], A[8, 8], b[8] = Kθ, 1, Kθ * θIsp
        # Kw indoor humidity ratio controller
        A[9, 5], A[9, 9], b[9] = Kw, 1, Kw * psy.w(θIsp, φIsp)
        x = np.linalg.solve(A, b)
        return x

    def solve_lin(self, θs0):
        """
        Finds saturation point on saturation curve ws = f(θs).
        Solves iterativelly lin_model(θs0); θs -> θs0 until ws between two
        iterations is practically the same.

        Parameters
        ----------
        ts0     initial guess saturation temperature

        Method from object
        ---------------------
        self.lin_model(ts0)

        Returns (10 variables, solution of lin_model)
        -------
        x : θs, ws, θS, wS, θI, wI, QsCC, QlCC, QtCC, QsHC
        """
        Δ_ws = 10e-3  # initial difference to start the iterations
        while Δ_ws > 0.01e-3:
            x = self.lin_model(θs0)
            Δ_ws = abs(psy.w(x[0], 1) - x[1])     # test convergence
            θs0 = x[0]                              # actualize ws
        return x

    def solve_ls(self, θSsp):
        """
        Controls supply temperature θS by d.a. mass flow rate m
        (finds m which solves θS = θSsp)
        Uses scipy.optimize.least_squares to solve the non-linear system

        Parameters
        ----------
        θSsp    saturation temperature set-point

        Returns (11 Unknowns)
        --------------------
        x = [ts, ws, tS, wS, tI, wI, QsCC, QlCC, QtCC, QsHC]
            given by self.solve_lin for m found by least_squares
        m -> chanhned in self.actual
         out        s              S          I
        ==0==>[CC]==1==>[HC]===F===2===>[TZ]==3==>
               /\\      /     /    |    //    ||
              t  sl    s     m     |   sl     ||
              |        |     |     |          ||
              |        |     |<-ls-|<-θS      ||
              |        |<------[K]------------|<-wI
              |<---------------[K]------------|<-θI

        Inputs:
        θo, φo      outdoor temperature & relative humidity
        θI, φI      indoor air temperature and humidity
        θS          supply air temperature
        QsTZ        sensible heat load of TZ
        QlTZ        latent heat load of TZ

        Elements (11 equations):
        CC          cooling coil (4 equations)
        HC          heating coil (2 equations)
        TZ          thermal zone (2 equations)
        F           fan (m is given)
        KθI         indoor temperature controller (1 equation)
        KwI         indoor humidity controller (1 equation)
        lsθS        mass flow rate of dry air controller (1 non-linear eq.)

        Outputs (11 unknowns):
        x:
        0, 1, 2     temperature and humidity ratio (6 unknowns)
        QsCC, QlCC  sensible and latent heat of CC (2 unknowns)
        QtCC        total heat load of CC (1 unknown)
        QsHC        sensible heat load of HC (1 unknown)
        Parameter:
        self.actual[0]:
        m           mass flow rate of dry air (1 unknown)

        """
        from scipy.optimize import least_squares

        def supply_air(m):
            """
            Uses scipy.opt.least_squares to find m which makes θS = θSsp
            Gives difference (tS - tSsp) function of m
                θS  calculated by self.solve_lin(ts0)
                m   bounds=(0, m_max); m_max hard coded in cool_05.py

            Parameters
            ----------
            m : mass flow rate of dry air

            From object
                Method: self.solve.lin(ts0)
                Variables: self.actual <- m (used in self.solve.lin)

            Returns
            -------
            tS - tSsp: difference between supply temp. and its set point
            """
            self.actual[0] = m
            x = self.solve_lin(θs_0)
            θS = x[2]       # supply air
            return abs(θSsp - θS)

        m = self.actual[0]
        res = least_squares(supply_air, m, bounds=(0, m_max))
        # gives m for tSsp; ts_0 is the initial guess of ts
        if res.cost < 0.1e-3:
            m = float(res.x)
            print(f'm = {m: 5.3f} kg/s')
        else:
            print('RecAirVAV: No solution for m')

        x = self.solve_lin(θs_0)
        self.actual[0] = m
        return x

    def psy_chart(self, x, θo, φo):
        """
        Plot results on psychrometric chart.

        Parameters
        ----------
        x = [ts, ws, tS, wS, tI, wI, QsCC, QlCC, QtCC, QsHC]
                    results of self.solve_lin or self.solve_ls
        θo, φo    outdoor point

        Returns
        -------
        None.

        """

        # Processes on psychrometric chart
        wO = psy.w(θo, φo)
        # Points: O, s, S, I
        θ = np.append(θo, x[0:6:2])
        w = np.append(wO, x[1:6:2])
        # Points       O   s   S   I     Elements
        A = np.array([[-1, 1, 0, 0],    # CC
                      [0, -1, 1, 0],    # HC
                      [0, 0, -1, 1]])   # TZ
        psy.chartA(θ, w, A)

        θ = pd.Series(θ)
        w = 1000 * pd.Series(w)         # kg/kg -> g/kg
        P = pd.concat([θ, w], axis=1)   # points
        P.columns = ['θ [°C]', 'w [g/kg]']

        output = P.to_string(formatters={
            'θ [°C]': '{:,.2f}'.format,
            'w [g/kg]': '{:,.2f}'.format})
        print()
        print(output)

        Q = pd.Series(x[6:], index=['QsCC', 'QlCC', 'QtCC', 'QsHC'])
        # Q.columns = ['kW']
        pd.options.display.float_format = '{:,.2f}'.format
        print()
        print(Q.to_frame().T / 1000, 'kW')
        return None

    def CAV_wd(self, m=3.333, θo=32, φo=0.5, θIsp=24, φIsp=0.5,
               QsTZ=20, QlTZ=15):
        """
        Constant air volume (CAV) to be used in Jupyter with widgets

        Parameters: given in Jupyetr widget
        ----------
        m, θo, φo, tIsp, φIsp: from self
        QsTZ, QlTZ : kW, sensible, latent load of thermal zone
                    given by widgets in Jupyter Lab

        Returns
        -------
        None.

         out        s              S          I
        ==0==>[CC]==1==>[HC]===F===2===>[TZ]==3==>
               /\\      /     /        / /    ||
              t  sl    s     m        s l     ||
              |        |                      ||
              |        |                      ||
              |        |<------[K]------------||<-wI<-φI
              |<---------------[K]------------|<-tI

        System:
            CC:     cooling coil (QtCC)
            HC:     heating coil (QsHC)
            TZ:     thermal zone (QsTZ, QlTZ)
            Kt:     controls indoor air temperature tI; commands QtCC
            Kw:     controls indoor humidity ration wI; commands QsHC
        """
        # To use fewer variables in Jupyter widget:
        # select what to be updated in self.actual, e.g.:
        # self.actual[[0, 1, 2, 5, 6]] = m, θo, φo, 1000 * QsTZ, 1000 * QlTZ

        self.actual = np.array([m, θo, φo, θIsp, φIsp,
                                1000 * QsTZ, 1000 * QlTZ])
        θ0 = 40
        x = self.solve_lin(θ0)
        print(f'm = {self.actual[0]: 5.3f} kg/s')
        self.psy_chart(x, self.actual[1], self.actual[2])

    def VAV_tS_wd(self, θSsp=18, θo=32, φo=0.5, θIsp=24, φIsp=0.5,
                  QsTZ=20, QlTZ=15):
        """
        Variable air volume (VAV) to be used in Jupyter with widgets

        Parameters
        ----------
        tSsp :          supply air temperature set point
        m, θo, φo, tIsp, φIsp: from self
        QsTZ, QlTZ : kW, sensible, latent load of thermal zone
                    given by widgets in Jupyter Lab

        Returns
        -------
        None.


         out        s              S          I
        ==0==>[CC]==1==>[HC]===F===2===>[TZ]==3==>
               /\\      /     /    |     //   ||
              t  sl    s     m     |    sl    ||
              |        |     |     |          ||
              |        |     |<-ls-|<-tS      ||
              |        |<------[K]------------|<-wI<-φI
              |<---------------[K]------------|<-tI

        System:
        CC:     cooling coil (QtCC)
        HC:     heating coil (QsHC)
        TZ:     thermal zone (QsTZ, QlTZ)
        F:      fan (m)
        KθI:    controls indoor air temperature θI; commands QtCC
        KwI:    controls indoor humidity ration wI; commands QsHC
        ls:     controls supply air temperature tS; commands m (used in VAV)
        """
        # Design values
        self.actual[1:] = θo, φo, θIsp, φIsp, 1000 * QsTZ, 1000 * QlTZ

        x = self.solve_ls(θSsp)
        self.psy_chart(x, θo, φo)


# # TESTS: uncomment
# # Create object
# m = 3.333
# θo = 32
# φo = 0.5
# tIsp = 24
# φIsp = 0.5
# QsTZ = 20   # kW
# QlTZ = 15   # kW
# CC = CcTz(m, θo, φo, tIsp, φIsp, 1e3 * QsTZ, 1e3 * QlTZ)

# # CAV
# print('\nCAV')
# CC.CAV_wd(m=3.333, θo=32, φo=0.5, θIsp=24, φIsp=0.5, QsTZ=20, QlTZ=15)
# # VAV
# print('\nVAV')
# CC.VAV_tS_wd(θSsp=18, θo=32, φo=0.5, θIsp=24, φIsp=0.5, QsTZ=20, QlTZ=15)
