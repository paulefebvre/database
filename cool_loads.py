#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 21:27:06 2021

@author: cghiaus

# Loads

Estimate the values of building characteristic and auxiliary loads

Total:
 - sensible: 45 kW
 - latent: 15 kW

**References**
Pennycook, K. (Ed.). (2003). Rules of Thumb: Guidelines for Building Services.
    BSRIA.
"""

import psychro as psy
# constants
c = 1e3         # J/kg K, air specific heat
l = 2496e3      # J/kg, latent heat
ρ = 1.2         # kg/m3, density

# Buildig dimensions
length = 20     # m
width = 30      # m
height = 3.5    # m
persons = 100   # m

sens_heat_person = 60       # W / person
latent_heat_person = 40     # W / person
load_m2 = 20        # W/m2
solar_m2 = 150      # W/m2 of window area
ACH = 1             # Air Cnhnages per Hour
U_wall = 0.4        # W/K, overall heat transfer coeff. walls
U_window = 3.5      # W/K, overall heat transfer coeff. windows

θo, φo = 32, 0.5    # outdoor temperature & relative humidity
θI, φI = 26, 0.5    # indoor temperature & relative humidity
wo = psy.w(θo, φo)
wI = psy.w(θI, φI)

floor_area = length * width
surface_envelope = 2 * (length + width) * height + floor_area
surface_wall = 0.9 * surface_envelope
surface_window = surface_envelope - surface_wall

# building conductance, W/K
UA = U_wall * surface_wall + U_window * surface_window

# infiltration mass flow rate, kg/s
mi = ACH * surface_envelope * height / 3600 * ρ

# gains, W
solar_gains = solar_m2 * surface_window
electrical_load = load_m2 * floor_area
Qsa = persons * sens_heat_person + solar_gains + electrical_load
Qla = persons * latent_heat_person

# thermal loads, W
QsTZ = (UA + mi * c) * (θo - θI) + Qsa  # sensible
QlTZ = mi * l * (wo - wI) + Qla         # latent

θS = θI - 15                            # °C supply air temperature
m = QsTZ / c / ((θI - θS))              # kg/s supply air mass flow rate

print(f'QsTZ = {QsTZ:.0f} W, QlTZ = {QlTZ:.0f} W')
print(f'UA = {UA:.0f} W/K, mi = {mi:.2f} kg/s,\
      Qsa = {Qsa:.0f} W, Qla = {Qla:.0f} W')
print(f'm = {m:.3f} kg/s')
