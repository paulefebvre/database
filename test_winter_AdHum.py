#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:37:10 2020

@author: cghiaus
test winter_AdHum.py
https://www.spyder-ide.org/blog/introducing-unittest-plugin/
"""
import winter_AdHum as ah
import numpy as np


def test_ahModelRecAir():
    """
     <----|<------------------------------------------|
          |                                           |
          |             |-------|                     |
    -o->MX1--0->HC1--1->|       MX2--3->HC2--4->TZ--5-|
                /       |       |        /      ||    |
                |       |->AH-2-|        |      BL    |
                |                        |            |
                |                        |<-----Kt----|<-t5
                |<------------------------------Kw----|<-w5
    """
    # Input data
    m = 4.9334
    alpha = 1
    beta = 0.1
    tS = 30
    tIsp, phiIsp = 18, 0.49
    tO, phiO = -1, 1
    Qsa, Qla = 0, 0
    mi = 2.18
    UA = 935.83

    # Expected output
    t0, w0 = -1,     3.5076e-03
    t1, w1 = 21.2,   3.5076e-03
    t2, w2 = 10.397, 7.8380e-03
    t3, w3 = 11.478, 7.4053e-03
    t4, w4 = 30,      7.4053e-03
    t5, w5 = 18,     6.2108e-03
    QHC1, QHC2 = 109556.66,  91374.67
    QsTZ, QlTZ = -59200.74, -14709.05
    y = [t0, w0, t1, w1, t2, w2, t3, w3, t4, w4, t5, w5,
         QHC1, QHC2, QsTZ, QlTZ]

    # Compare results and expected
    np.testing.assert_almost_equal(
        ah.ModelRecAir(m, alpha, beta, tS, tIsp, phiIsp, tO, phiO,
                       Qsa, Qla, mi, UA),
        y, 2)
