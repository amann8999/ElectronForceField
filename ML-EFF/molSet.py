# Creates molName: a list of different sets of molecule names

molName = []

# 0: All atoms (18)
molName.append(['H1', 'He1', 'Li1', 'Be1', 'B1', 'C1', 'N1', 'O1', 'F1', 'Ne1', \
                'Na1', 'Mg1', 'Al1', 'Si1', 'P1', 'S1', 'Cl1', 'Ar1'])

# 1: Minimum optimization set (6 atoms + 6 molecules)
molName.append(['H1', 'Li1', 'C1', 'N1', 'O1', 'F1', \
                'H2', 'Li2', 'CH4', 'NH3', 'H2O', 'HF'])

# 2: Larger set of 1st and 2nd period elements (6 atoms + 18 molecules)
molName.append(['H1', 'Li1', 'C1', 'N1', 'O1', 'F1', \
                'H2', 'Li2', 'LiH', 'N2', 'O2s', 'O2', 'F2', 'CH4', \
                'NH3', 'H2O', 'HF', \
                'CO', 'CO2', 'HCN', 'C2H4', 'C2H6', 'H3COH', 'C6H6'])

# 3: Full set of 1st and 2nd period elements (40 molecules/atoms/ions)
molName.append(['H1', 'He1', 'Li1', 'C1', 'N1', 'O1', 'F1', \
                'H2', 'Li2', 'LiH', \
                'CH', 'CH2s', 'CH2t', 'CH3', 'CH4', 'C2H4', 'C2H6', 'C6H6', 'C10H8', \
                'N2', 'NH', 'NH2', 'NH3', 'HCN', \
                'O2', 'O2s', 'OH', 'H2O', 'CO', 'CO2', 'H3COH', \
                'F2', 'HF', 'LiF', \
                'He1+', 'H2+', 'HeH+', 'Li2+', 'NH4+', 'OH-'])

# 4: Set of 1st, 2nd, and 3rd period elements (72 molecules/atoms/ions)
molName.append(['H1', 'He1', 'Li1', 'C1', 'N1', 'O1', 'F1', \
                'Na1', 'Si1', 'P1', 'S1', 'Cl1', \
                'H2', 'Li2', 'LiH', \
                'CH', 'CH2s', 'CH2t', 'CH3', 'CH4', 'C2H4', 'C2H6', 'C6H6', \
                'N2', 'NH', 'NH2', 'NH3', 'HCN', \
                'O2', 'O2s', 'OH', 'H2O', 'CO', 'CO2', 'H3COH', \
                'F2', 'HF', 'LiF', \
                'SiH2s', 'SiH2t', 'SiH4', 'P2', 'PH3', \
                'H2S', 'H3CSH', \
                'Cl2', 'HCl', 'ClF', 'NaCl', 'CH3Cl', 'HOCl', \
                'H1+', 'He1+', 'Li1+', 'C1+', 'N1+', 'O1+', 'F1+', \
                'Na1+', 'Si1+', 'P1+', 'S1+', 'Cl1+', \
                'H2+', 'HeH+', 'CH+', 'NH4+', \
                'O1-', 'F1-', 'Cl1-', 'NH2-', 'OH-'])

# 5: Extended 127-set with all elements up to Ar and some singlet states
molName.append(['H1', 'He1', 'Li1', 'Be1', 'B1', 'C1', 'C1s', \
                'N1', 'N1d', 'O1', 'O1s', 'F1', 'Ne1', \
                'Na1', 'Mg1', 'Al1', 'Si1', 'Si1s', \
                'P1', 'P1d', 'S1', 'S1s', 'Cl1', 'Ar1', \
                'H2', 'Li2', 'Be2', 'B2', 'C2', 'N2', 'O2', 'O2s', 'F2', \
                'Na2', 'Mg2', 'P2', 'Cl2', \
                'LiH', 'BeH', 'BH', 'CH', 'NH', 'NHs', 'OH', 'HF', \
                'NaH', 'MgH', 'AlH', 'SiH', 'PH', 'HS', 'HCl', \
                'BeH2', 'BH2', 'BH3', \
                'CH2s', 'CH2t', 'CH3', 'CH4', 'C2H4', 'C2H6', \
                'NH2', 'NH3', 'H2O', \
                'MgH2', 'AlH2', 'AlH3', 'SiH2s', 'SiH2t', 'SiH3', 'SiH4', \
                'PH3', 'H2S', \
                'BeO', 'CO', 'CO2', 'BO', 'NO', 'FO', \
                'MgO', 'AlO', 'SiO', 'SiO2',  'PO', 'SO', 'ClO', \
                'LiF', 'ClF', 'NaCl', 'HCN', 'H3COH', 'H3CSH', 'CH3Cl', 'HOCl', \
                'H1+', 'He1+', \
                'Li1+', 'Be1+', 'B1+', 'C1+', 'N1+', 'O1+', 'F1+', 'Ne1+', \
                'Na1+', 'Mg1+', 'Al1+', 'Si1+', 'P1+', 'S1+', 'Cl1+', 'Ar1+', \
                'H2+', 'HeH+', 'LiH+', 'BeH+', 'CH+', 'CH3+', 'NH4+', \
                'MgH+', 'SiH+', 'ArH+', \
                'O1-', 'F1-', 'Cl1-', 'NH2-', 'OH-'])

# 6: NIST Set (568 atoms + molecules)
molName.append(['H1', 'F1', 'OH', 'He1+', 'H2+', \
                'H2', 'LiH', 'BeH', 'BH', 'CH', 'HF', 'NaH', 'MgH',  \
                'AlH', 'SiH', 'PH', 'HS', 'HCl', 'BeH2', 'BH2', 'CH2s', 'CH2t', 'NH2',  \
                'HNB', 'HCN', 'H2O', 'BeOH', 'HBO', 'COH', 'NOH', 'HO2', 'HOF',  \
                'MgH2', 'AlH2', 'SiH2s', 'SiH2t', 'HCP', 'HPO', 'H2S', 'H3CSH', 'HBS', 'HNS',  \
                'SOH', 'HS2', 'HCCl', 'HOCl', 'HSiCl', 'BH3', 'HBBH', 'CH3', 'H3COH', 'C2H2',  \
                'NH3', 'HBNH', 'H2CN', 'N2H2', 'HCCN', 'HN3', 'H2CO', 'H2NO', 'H2O2', 'HCCO',  \
                'HOCO', 'HNO2', 'CH2F', 'NH2F', 'HCCF', 'BHF2', 'CHF2', 'NHF2', 'AlH3',  \
                'SiH3', 'Si2H2', 'H2CS', 'HOSH', 'HSCN', 'H2S2', 'NH2Cl', 'HCCCl', 'CHFCl', 'HOClO',  \
                'BHCl2', 'CHCl2', 'CH2Cl', 'CH4', 'CH3Li', 'C2H3', 'CH2NH', 'HNCNH', 'HCCCN',  \
                'NH2OH', 'HCOOH', 'NH2NO', 'HNO3', 'CH3F', 'CH2F2', 'CHF3', 'SiH4',  \
                'H3PO', 'NH2SH', 'HSSSH', 'CH3Cl', 'CH2FCl', 'CH2Cl2', 'CHF2Cl', 'HClO3', 'SiH2Cl2',  \
                'CHCl3', 'B2H4', 'C2H4', 'C4H2', 'BH2NH2', 'N2H4', 'CHNCH2', 'HNCCNH', 'H2OH2O',  \
                'CH2BOH', 'CH3CO', 'CHONH2', 'CH3OO', 'C2H2O2', 'H2CO3', 'HOONO2', 'CH2CHF', 'CH2FOH', 'CH2CF2',  \
                'CF3OH', 'CH2SiH2', 'SiH3OH', 'Si2H4', 'P2H4', 'CHSNH2', 'NH4Cl', 'C2H3Cl', 'CHFCHCl', 'CH2CCl2',  \
                'HClO4', 'HeH+', 'LiH+', 'BeH+', 'CH+', 'OH+', 'HF+', 'MgH+',  \
                'SiH+', 'HS+', 'HCl+', 'ArH+', 'BeH2+', 'BH2+', 'CH2+', 'C2H+', 'NH2+', 'HNC+',  \
                'NNH+', 'H2O+', 'HCO+', 'HNO+', 'HO2+', 'H2F+', 'F2H+', 'AlH2+', 'SiH2+', 'PH2+',  \
                'HCP+', 'H2S+', 'HSO+', 'H2Cl+', 'HOCl+', 'ClFH+', 'Cl2H+', 'CH3+', 'C2H2+', 'H2CN+',  \
                'N2H2+', 'HN3+', 'H3O+', 'H2CO+', 'H2O2+', 'HOCO+', 'NNOH+', 'O3H+', 'CH2F+', 'HFCO+',  \
                'CHF2+', 'SiH3+', 'H3S+', 'H2CS+', 'SO2H+', 'H2S2+', 'CH2Cl+', 'CHFCl+', 'CHCl2+', 'C2H3+',  \
                'NH4+', 'CH2NH+', 'NH2NN+', 'CH3O+', 'NH2OH+', 'H3O2+', 'H2COO+', 'CH3F+', 'NF3H+', 'PH4+',  \
                'H2CSH+', 'C4H2+', 'N2H4+', 'CH3CN+', 'NCNH3+', 'CH3OH+', 'CH3CO+', 'CH3OO+', 'SiH5+',  \
                'CH3SH+', 'H1-', 'BeH-', 'OH-', 'SiH-', 'PH-', 'HS-', 'BH2-', 'CH2-', 'C2H-',  \
                'HCO-', 'AlH2-', 'SiH2-', 'PH2-', 'HCS-', 'HNS-', 'CH3-', 'HCCH-', 'H2CO-',  \
                'HCO2-', 'SiH3-', 'H2CS-', 'BH4-', 'C2H3-', 'CH3O-', 'AlH4-', 'CH3OO-', 'He1',  \
                'Li1', 'Li2', 'LiBe', 'LiN', 'LiO', 'LiF', 'LiMg',  \
                'LiAl', 'LiS', 'LiCl', 'Li2S', 'LiF+', 'LiNe+', 'Li2O+', 'LiO-',  \
                'Be1', 'Be2', 'BeB', 'BeN', 'BeO', 'BeF', 'NaBe', 'BeMg', 'BeS', 'BeCl',  \
                'BeF2', 'BeCl2', 'BeCO3', 'Be1+', 'BeN+', 'BeO+', 'BeF+', 'BeCl+', 'Be1-', 'BeN-',  \
                'BeF-', 'BeCl-', 'B1', 'B2', 'BN', 'BO', 'BAl', 'BSi', 'BP',  \
                'BS', 'BCl', 'BO2', 'BF2', 'B2O2', 'BF3', 'BClF2', 'BCl3', 'B2O3', 'B1+',  \
                'B2+', 'BO+', 'BS+', 'BO2+', 'BO-', 'BS-', 'BO2-', 'BF4-',  \
                'C1', 'C2', 'CN', 'CO', 'CF', 'SiC', 'CS', 'CCl', 'NCN',  \
                'CCO', 'NCO', 'CO2', 'FCN', 'CF2', 'AlCN', 'OCS', 'SCN', 'CS2',  \
                'ClCO', 'CFCl', 'CCl2', 'C4', 'C2N2', 'C3O', 'CO3', 'C2F2', 'CF2O', 'CF3',  \
                'COFCl', 'CF2Cl', 'C2Cl2', 'CCl2O', 'CFCl2', 'Cl2CS', 'CCl3', 'C3O2', 'CF3Cl', 'CF2Cl2',  \
                'CCl4', 'C4F2', 'CF3OF', 'C2ClF3', 'ClCOClCO', 'CF2CCl2', 'C1+', 'CN+', 'CO+',  \
                'CF+', 'CP+', 'CS+', 'CCl+', 'CO2+', 'FCO+', 'OCS+', 'CS2+', 'CCl3+', 'C1-',  \
                'CN-', 'CO-', 'SiC-', 'CP-', 'CS-', 'NCO-', 'CO2-', 'SCN-', 'CS2-', 'N1',  \
                'N2', 'NH', 'NO', 'NF', 'MgN', 'AlN', 'PN', 'NS', 'NCl', 'N3',  \
                'N2O', 'NO2', 'NF2', 'NNS', 'SNO', 'FNS', 'ClNO', 'FNO2', 'N2F2', 'S2N2',  \
                'ClNO2', 'NCl3', 'N2O4', 'N3P3', 'N1+', 'N2+', 'NO+', 'SiN+', 'PN+', 'NS+',  \
                'NCl+', 'N3+', 'NO2+', 'NF2+', 'NO3+', 'N1-', 'N2-', 'NH2-', 'SiN-', 'PN-',  \
                'NS-', 'N3-', 'NO2-', 'NF3-', 'O1', 'O2', 'O2s', 'FO', 'NaO', 'MgO',  \
                'AlO', 'SiO', 'PO', 'SO', 'ClO', 'O3', 'F2O', 'SiO2', 'PPO', 'SO2',  \
                'FClO', 'OClO', 'OPCl', 'Cl2O', 'SO3', 'ClO2F', 'ClOOCl', 'SO2F2',  \
                'SOF4', 'O1+', 'O2+', 'AlO+', 'SiO+', 'PO+', 'SO+', 'ClO+', 'FOO+',  \
                'PO2+', 'SO2+', 'OClO+', 'SO3+', 'O2-', 'FO-', 'NaO-', 'AlO-', 'SiO-',  \
                'PO-', 'SO-', 'ClO-', 'PO2-', 'SO2-', 'OClO-', 'SO3-', 'ClO3-', 'PO4---', 'SO4--',  \
                'ClO4-', 'F2', 'NaF', 'AlF', 'SiF', 'SF', 'ClF', 'MgF2', 'SiF2',  \
                'PF2', 'SF2', 'SFCl', 'SiF3', 'PF3', 'SF3', 'FSSF', 'ClF3',  \
                'F1+', 'F2+', 'MgF+', 'SiF+', 'PF+', 'SF+', 'SiF2+', 'PF2+', 'SiF3+',  \
                'SF5+', 'F1-', 'SiF-', 'PF-', 'SF-', 'F3-', 'SF2-', 'SiF5-', 'Ne1', 'Ne1+',  \
                'Ne2+', 'Na1', 'NaMg', 'NaAl', 'NaS', 'NaCl', 'NaCl+', 'Mg1',  \
                'Mg2', 'MgS', 'MgCl2', 'Mg1+', 'Mg2+', 'MgS+', 'MgCl+', 'MgS-', 'MgCl-', 'Al1',  \
                'AlP', 'AlS', 'AlCl', 'AlCl3', 'Al1+', 'AlS+', 'Si1', 'SiP', 'SiS',  \
                'SiCl', 'SiS2', 'Si1+', 'SiP+', 'SiS+', 'Si1-', 'Si2-', 'SiP-', 'P1',  \
                'P2', 'PH3', 'PS', 'P4', 'PCl3', 'P1+', 'PS+', 'PCl+', 'P1-', 'P2-',  \
                'PS-', 'PCl-', 'S1', 'S2', 'S3', 'SCl2', 'S4', 'ClSSCl',  \
                'S1+', 'SCl+', 'S2-', 'SCl-', 'S3-', 'Cl1', 'Cl2', 'Cl1+',  \
                'Cl2+', 'Cl1-', 'Ar1'])
