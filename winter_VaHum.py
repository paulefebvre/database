"""
Tue Apr 21 11:36:00 2020
Heating and vapor humidification
Two HVAC systems: all out air and recirculated air

Building characterised by:
    UA      global conductivity
    mi      air infiltration mass flow
    Qsa     auxiliary sensible loads
    Qla     auxiliary latent loads
Indoor conditions
    tI      indoor temperature
    phiI    indoor relative humidity
Outdoor conditions
    tO      outdoor temperature
    phiO    outdoor relative humidity
Constant air volume (CAV)
    m       mass flow rate constant (cf. design conditions)
    tS      supply air temperature variable
Variable air volume (VAV)
    tS      supply air temperature controlled
    m       mass flow rate varable

Inputs given in Jupyter Notebook

"""
import numpy as np
import pandas as pd
import psychro as psy
import matplotlib.pyplot as plt

tOd = -1                    # outdoor design conditions
mid = 2.18                  # infiltration design

# constants
c = 1e3                     # air specific heat J/kg K
l = 2496e3                  # latent heat J/kg

# *****************************************
# ALL OUT AIR
# *****************************************


def ModelAllOutAir(m, tS, tIsp, phiIsp, tO, phiO, Qsa, Qla, mi, UA):
    """
    Model:
        All outdoor air
        CAV Constant Air Volume:
            mass flow rate given
            control of indoor condition (t2, w2)
    INPUTS:
        m     mass flow of supply dry air kg/s
        tS    supply air °C
        tIsp  indoor air setpoint °C
        phiIsp -
        tO    outdoor temperature for design °C
        phiO  outdoor relative humidity for design -
        Qsa   aux. sensible heat W
        Qla   aux. latente heat W
        mi    infiltration massflow rate kg/s
        UA    global conductivity bldg W/K

    OUTPUTS:
        x     vector 10 elements:
            t0, w0, t1, w1, t2, w2, QsHC, QlVH, QsTZ, QlTZ

    System:
        HC:     Heating Coil
        VH:     Vapor Humidifier
        TZ:     Thermal Zone
        BL:     Building
        Kw:     Controller - humidity
        Kt:     Controller - temperature
        o:      outdoor conditions
    
    10 Unknowns
        0, 1, 2 points (temperature, humidity ratio)
        QsHC, QlVH, QsTZ, QlTZ

    --o->HC--0->VH--1->TZ--2-->
         |       |     ||  |
         |       |     BL  |
         |       |         |
         |       |<----Kw--|-w2
         |<------------Kt--|-t2
    """
    Kt, Kw = 1e10, 1e10             # controller gain
    wO = psy.w(tO, phiO)            # outdoor mumidity ratio
    wIsp = psy.w(tIsp, phiIsp)      # indoor mumidity ratio

    # Model
    A = np.zeros((10, 10))          # coefficents of unknowns
    b = np.zeros(10)                # vector of inputs
    # HC heating coil
    A[0, 0], A[0, 6],           b[0] = m*c,     -1,     m*c*tO
    A[1, 1],                    b[1] = m*l,     m*l*wO
    # VA vapor humidifier
    A[2, 0], A[2, 2],           b[2] = -m*c,    m*c,    0
    A[3, 1], A[3, 3], A[3, 7],  b[3] = -m*l,    m*l,    -1,     0
    # TZ thermal zone
    A[4, 2], A[4, 4], A[4, 8],  b[4] = -m*c,    m*c,    -1,     0
    A[5, 3], A[5, 5], A[5, 9],  b[5] = -m*l,    m*l,    -1,     0
    # BL building
    A[6, 4], A[6, 8],           b[6] = UA+mi*c, 1,  (UA + mi*c)*tO + Qsa
    A[7, 5], A[7, 9],           b[7] = mi*l,    1,  mi*l*wO + Qla
    # Kt indoor temperature controller
    A[8, 4], A[8, 8],           b[8] = Kt,      1,  Kt*tIsp
    # Kw indoor hum.ratio controller
    A[9, 5], A[9, 9],           b[9] = Kw,      1,  Kw*wIsp

    # Solution
    x = np.linalg.solve(A, b)
    return x


def AllOutAirCAV(tS=30, tIsp=18, phiIsp=0.5, tO=-1, phiO=1,
                 Qsa=0, Qla=0, mi=2.12, UA=935.83):
    """
    All out air
    CAV Constant Air Volume:
        mass flow rate calculated for design conditions
        maintained constant in all situations

    INPUTS:
        m     mass flow of supply dry air kg/s
        tS    supply air °C
        tIsp  indoor air setpoint °C
        phiIsp -
        tO    outdoor temperature for design °C
        phiO  outdoor relative humidity for design -
        Qsa   aux. sensible heat W
        Qla   aux. latente heat W
        mi    infiltration massflow rate kg/s
        UA    global conductivity bldg W/K

    System:
        HC:     Heating Coil
        VH:     Vapor Humidifier
        TZ:     Thermal Zone
        BL:     Building
        Kw:     Controller - humidity
        Kt:     Controller - temperature
        o:      outdoor conditions

    10 Unknowns
        0, 1, 2 points (temperature, humidity ratio)
        QsHC, QlVH, QsTZ, QlTZ

    --o->HC--0->VH--1->TZ--2-->
         /       /     ||  |
         |       |     BL  |
         |       |         |
         |       |<----Kw--|-w2
         |<------------Kt--|-t2
    """
    plt.close('all')
    wO = psy.w(tO, phiO)            # hum. out

    # Mass flow rate for design conditions
    # tOd = -1                        # outdoor design conditions
    # mid = 2.18                      # infiltration design
    QsZ = UA*(tOd - tIsp) + mid*c*(tOd - tIsp)
    m = - QsZ/(c*(tS - tIsp))
    print('Winter All_out_air CAV')
    print(f'm = {m: 5.3f} kg/s constant (from design conditions)')
    print(f'Design conditions tS = {tS: 3.1f} °C,'
          f'mi = {mid:3.1f} kg/s, tO = {tOd:3.1f} °C, '
          f'tI = {tIsp:3.1f} °C')

    # Model
    x = ModelAllOutAir(m, tS, tIsp, phiIsp, tO, phiO, Qsa, Qla, mi, UA)

    # Processes on psychrometric chart
    t = np.append(tO, x[0:5:2])
    w = np.append(wO, x[1:6:2])

    # Adjancy matrix: rows=lines; columns=points
    # Points       O    0   1   2       Elements
    A = np.array([[-1,  1,  0,  0],     # HC
                 [0,  -1,  1,  0],      # VH
                 [0,   0,  1, -1]])     # TZ

    psy.chartA(t, w, A)

    t = pd.Series(t)
    w = 1000*pd.Series(w)
    P = pd.concat([t, w], axis=1)       # points
    P.columns = ['t [°C]', 'w [g/kg]']

    output = P.to_string(formatters={
        't [°C]': '{:,.2f}'.format,
        'w [g/kg]': '{:,.2f}'.format
    })
    print()
    print(output)

    Q = pd.Series(x[6:], index=['QsHC', 'QlVH', 'QsTZ', 'QlTZ'])
    # Q.columns = ['kW']
    pd.options.display.float_format = '{:,.2f}'.format
    print()
    print(Q.to_frame().T/1000, 'kW')

    return x


def AllOutAirVAV(tSsp=30, tIsp=18, phiIsp=0.5, tO=-1, phiO=1,
                 Qsa=0, Qla=0, mi=2.12, UA=935.83):
    """
    All out air
    Heating & Vapor humidification
    VAV Variable Air Volume:
        mass flow rate calculated to have const. supply temp.

    INPUTS:
        tS    supply air °C
        tIsp  indoor air setpoint °C
        phiIsp -
        tO    outdoor temperature for design °C
        phiO  outdoor relative humidity for design -
        Qsa   aux. sensible heat W
        Qla   aux. latente heat W
        mi    infiltration massflow rate kg/s
        UA    global conductivity bldg W/K

    System:
        HC:     Heating Coil
        VH:     Vapor Humidifier
        TZ:     Thermal Zone
        BL:     Building
        Kw:     Controller - humidity
        Kt:     Controller - temperature
        o:      outdoor conditions

    10 Unknowns
        0, 1, 2 points (temperature, humidity ratio)
        QsHC, QlVH, QsTZ, QlTZ

    --o->HC--0->VH--F-----1-->TZ--2-->
         /       /  |     |   ||  |
         |       |  |     |   BL  |
         |       |  |_Kt1_|       |
         |       |                |
         |       |<----Kw---------|-w2
         |<------------Kt---------|-t2

        Mass-flow rate (VAV) I-controller:
        start with m = 0
        measure the supply temperature
        while -(tSsp - tS)>0.01, increase m (I-controller)
    """
    plt.close('all')
    wO = psy.w(tO, phiO)            # outdoor mumidity ratio

    # Mass flow rate
    DtS, m = 2, 0                   # initial temp; diff; flow rate
    while DtS > 0.01:
        m = m + 0.01                # mass-flow rate I-controller
        # Model
        x = ModelAllOutAir(m, tSsp, tIsp, phiIsp, tO, phiO, Qsa, Qla, mi, UA)
        tS = x[2]
        DtS = -(tSsp - tS)
    print('Winter All_out_air VAV')
    print(f'm = {m: 5.3f} kg/s')
    # Processes on psychrometric chart
    t = np.append(tO, x[0:5:2])
    w = np.append(wO, x[1:6:2])
    # Points       o    0   1   2       Elements
    A = np.array([[-1,  1,  0,  0],     # HC
                 [0,  -1,  1,  0],      # VH
                 [0,   0,  1, -1]])     # TZ

    psy.chartA(t, w, A)

    t = pd.Series(t)
    w = 1000*pd.Series(w)
    P = pd.concat([t, w], axis=1)       # points
    P.columns = ['t [°C]', 'w [g/kg]']

    output = P.to_string(formatters={
        't [°C]': '{:,.2f}'.format,
        'w [g/kg]': '{:,.2f}'.format
    })
    print()
    print(output)

    Q = pd.Series(x[6:], index=['QsHC', 'QlVH', 'QsTZ', 'QlTZ'])
    # Q.columns = ['kW']
    pd.options.display.float_format = '{:,.2f}'.format
    print()
    print(Q.to_frame().T/1000, 'kW')

    return None


# *****************************************
# RECYCLED AIR
# *****************************************


def ModelRecAir(m, alpha, tS, tIsp, phiIsp, tO, phiO, Qsa, Qla, mi, UA):
    """
    Model:
        Heating and vapor humidification
        Recycled air
        CAV Constant Air Volume:
            mass flow rate calculated for design conditions
            maintained constant in all situations
    INPUTS:
        m     mass flow of supply dry air kg/s
        alpha mixing ratio of outdoor air
        tS    supply air °C
        tIsp  indoor air setpoint °C
        phiIsp -
        tO    outdoor temperature for design °C
        phiO  outdoor relative humidity for design -
        Qsa   aux. sensible heat W
        Qla   aux. latente heat W
        mi    infiltration massflow rate kg/s
        UA    global conductivity bldg W/K

    OUTPUTS:
        x     vector 12 elements:
            t0, w0, t1, w1, t2, w2, t3, w3, QsHC, QlVH, QsTZ, QlTZ

    System:
        MX:     Mixing Box
        HC:     Heating Coil
        VH:     Vapor Humidifier
        TZ:     Thermal Zone
        BL:     Buildings
        Kw:     Controller - humidity
        Kt:     Controller - temperature
        o:      outdoor conditions

    12 Unknowns
        0, 1, 2, 3 points (temperature, humidity ratio)
        QsHC, QlVH, QsTZ, QlTZ

    <-3--|<-------------------------|
         |                          |
    -o->MX--0->HC--1->VH--2->TZ--3-->
               /       /     ||  |
               |       |     BL  |
               |       |         |
               |       |<----Kw--|-w3
               |<------------Kt--|-t3
    """
    Kt, Kw = 1e10, 1e10             # controller gain
    wO = psy.w(tO, phiO)            # hum. out
    wIsp = psy.w(tIsp, phiIsp)      # hum. in set point

    # Model
    A = np.zeros((12, 12))          # coefficents of unknowns
    b = np.zeros(12)                # vector of inputs
    # MX mixing box
    A[0, 0], A[0, 6], b[0] = m*c, -(1 - alpha)*m*c, alpha*m*c*tO
    A[1, 1], A[1, 7], b[1] = m*l, -(1 - alpha)*m*l, alpha*m*l*wO
    # HC hearing coil
    A[2, 0], A[2, 2], A[2, 8], b[2] = m*c, -m*c, 1, 0
    A[3, 1], A[3, 3], b[3] = m*l, -m*l, 0
    # VH vapor humidifier
    A[4, 2], A[4, 4], b[4] = m*c, -m*c, 0
    A[5, 3], A[5, 5], A[5, 9], b[5] = m*l, -m*l, 1, 0
    # TZ thermal zone
    A[6, 4], A[6, 6], A[6, 10], b[6] = m*c, -m*c, 1, 0
    A[7, 5], A[7, 7], A[7, 11], b[7] = m*l, -m*l, 1, 0
    # BL building
    A[8, 6], A[8, 10], b[8] = (UA + mi*c), 1, (UA + mi*c)*tO + Qsa
    A[9, 7], A[9, 11], b[9] = mi*l, 1, mi*l*wO + Qla
    # Kt indoor temperature controller
    A[10, 6], A[10, 8], b[10] = Kt, 1, Kt*tIsp
    # Kw indoor humidity controller
    A[11, 7], A[11, 9], b[11] = Kw, 1, Kw*wIsp

    # Solution
    x = np.linalg.solve(A, b)
    return x


def RecAirCAV(alpha=0.5, tS=30, tIsp=18, phiIsp=0.5, tO=-1, phiO=1,
              Qsa=0, Qla=0, mi=2.12, UA=935.83):
    """
    CAV Constant Air Volume:
    mass flow rate calculated for design conditions
    maintained constant in all situations
    INPUTS:
        alpha mixing ratio of outdoor air
        tS    supply air °C
        tIsp  indoor air setpoint °C
        phiIsp -
        tO    outdoor temperature for design °C
        phiO  outdoor relative humidity for design -
        Qsa   aux. sensible heat W
        Qla   aux. latente heat W
        mi    infiltration massflow rate kg/s
        UA    global conductivity bldg W/K

    System:
        HC:     Heating Coil
        VH:     Vapor Humidifier
        TZ:     Thermal Zone
        Kw:     Controller - humidity
        Kt:     Controller - temperature
        o:      outdoor conditions

    12 Unknowns
        0, 1, 2, 3 points (temperature, humidity ratio)
        QsHC, QlVH, QsTZ, QlTZ

    <-3--|<-------------------------|
         |                          |
    -o->MX--0->HC--1->VH--2->TZ--3-->
               /       /     ||  |
               |       |     BL  |
               |       |         |
               |       |_____Kw__|_w3
               |_____________Kt__|_t3
    """
    plt.close('all')
    wO = psy.w(tO, phiO)            # hum. out

    # Mass flow rate for design conditions
    # Supplay air mass flow rate
    # QsZ = UA*(tO - tIsp) + mi*c*(tO - tIsp)
    # m = - QsZ/(c*(tS - tIsp)
    # where
    # tOd, wOd = -1, 3.5e-3           # outdoor
    # tS = 30                       # supply air
    # mid = 2.18                     # infiltration
    QsZ = UA*(tOd - tIsp) + mid*c*(-1 - tIsp)
    m = - QsZ/(c*(tS - tIsp))
    print('Winter Recirculated_air CAV')
    print(f'm = {m: 5.3f} kg/s constant (from design conditions)')
    print(f'Design conditions tS = {tS: 3.1f} °C,'
          f'mi = {mid:3.1f} kg/s, tO = {tOd:3.1f} °C, '
          f'tI = {tIsp:3.1f} °C')

    # Model
    x = ModelRecAir(m, alpha, tS, tIsp, phiIsp, tO, phiO, Qsa, Qla, mi, UA)
    # (m, tS, mi, tO, phiO, alpha)

    # Processes on psychrometric chart
    # Points      o    0    1   2   3       Elements
    A = np.array([[-1,  1,  0,  0, -1],     # MX
                 [0,  -1,  1,  0,   0],     # HC
                 [0,   0,  -1, 1,   0],     # VH
                 [0,   0,  0, -1,   1]])    # TZ
    t = np.append(tO, x[0:8:2])

    print(f'wO = {wO:6.5f}')
    w = np.append(wO, x[1:8:2])
    psy.chartA(t, w, A)

    t = pd.Series(t)
    w = 1000*pd.Series(w)
    P = pd.concat([t, w], axis=1)       # points
    P.columns = ['t [°C]', 'w [g/kg]']

    output = P.to_string(formatters={
        't [°C]': '{:,.2f}'.format,
        'w [g/kg]': '{:,.2f}'.format
    })
    print()
    print(output)

    Q = pd.Series(x[8:], index=['QsHC', 'QlVH', 'QsTZ', 'QlTZ'])
    pd.options.display.float_format = '{:,.2f}'.format
    print()
    print(Q.to_frame().T/1000, 'kW')

    return None


def RecAirVAV(alpha=0.5, tSsp=30, tIsp=18, phiIsp=0.5, tO=-1, phiO=1,
              Qsa=0, Qla=0, mi=2.12, UA=935.83):
    """
    CAV Variable Air Volume:
    mass flow rate calculated s.t.
    he supply temp. is maintained constant in all situations
    INPUTS:
    INPUTS:
        m     mass flow of supply dry air kg/s
        alpha mixing ratio of outdoor air
        tS    supply air °C
        tIsp  indoor air setpoint °C
        phiIsp -
        tO    outdoor temperature for design °C
        phiO  outdoor relative humidity for design -
        Qsa   aux. sensible heat W
        Qla   aux. latente heat W
        mi    infiltration massflow rate kg/s
        UA    global conductivity bldg W/K

    System (CAV & m introduced by the Fan is cotrolled by tS )
        HC:     Heating Coil
        VH:     Vapor Humidifier
        TZ:     Thermal Zone
        F:      Supply air fan
        Kw:     Controller - humidity
        Kt:     Controller - temperature
        o:      outdoor conditions

    12 Unknowns
        0, 1, 2, 3 points (temperature, humidity ratio)
        QsHC, QlVH, QsTZ, QlTZ

    <----|<--------------------------------|
         |                                 |
    -o->MX--0->HC--1->VH--F-----2-->TZ--3-->
               /       /  |     |   ||  |
               |       |  |     |   BL  |
               |       |  |     |       |
               |       |  |_Kt2_|_t2    |
               |       |                |
               |       |_____Kw_________|_w3
               |_____________Kt_________|_t3

    Mass-flow rate (VAV) I-controller:
        start with m = 0
        measure the supply temperature
        while -(tSsp - tS)>0.01, increase m (I-controller)
    """
    plt.close('all')
    wO = psy.w(tO, phiO)            # hum. out

    # Mass flow rate
    DtS, m = 2, 0                   # initial temp; diff; flow rate
    while DtS > 0.01:
        m = m + 0.01

        # Model
        x = ModelRecAir(m, alpha, tSsp, tIsp, phiIsp, tO, phiO,
                        Qsa, Qla, mi, UA)
        tS = x[4]
        DtS = -(tSsp - tS)

    print('Winter Rec_air VAV')
    print(f'm = {m: 5.3f} kg/s')

    # Processes on psychrometric chart
    # Points      o    0    1   2   3       Elements
    A = np.array([[-1,  1,  0,  0, -1],     # MX
                 [0,  -1,  1,  0,   0],     # HC
                 [0,   0,  -1, 1,   0],     # VH
                 [0,   0,  0, -1,   1]])    # TZ
    t = np.append(tO, x[0:8:2])
    print(f'wO = {wO:6.5f}')
    w = np.append(wO, x[1:8:2])
    psy.chartA(t, w, A)

    t = pd.Series(t)
    w = 1000*pd.Series(w)
    P = pd.concat([t, w], axis=1)           # points
    P.columns = ['t [°C]', 'w [g/kg]']

    output = P.to_string(formatters={
        't [°C]': '{:,.2f}'.format,
        'w [g/kg]': '{:,.2f}'.format
    })
    print()
    print(output)

    Q = pd.Series(x[8:], index=['QsHC', 'QlVH', 'QsTZ', 'QlTZ'])
    pd.options.display.float_format = '{:,.2f}'.format
    print()
    print(Q.to_frame().T/1000, 'kW')

    return None


# Uncomment to test a function
# AllOutAirCAV()
# AllOutAirVAV()
# RecAirCAV()
# RecAirVAV()
