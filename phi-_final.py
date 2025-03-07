import pandas as pd
import numpy as np


excel_file = pd.ExcelFile("phim.xlsx")

df = excel_file.parse("Sheet1")
x0 = excel_file.parse("x0")
y0 = excel_file.parse("y0")
z0 = excel_file.parse("z0")
lx0 = excel_file.parse("0x")
ly0 = excel_file.parse("0y")
lz0 = excel_file.parse("0z")

r_z = [x0, y0, z0] # data for i0
l_z = [lx0, ly0, lz0] # data for 0i

data = []
for index, row in df.iterrows():
    row_values = row.values.tolist()
    data.append(row_values)


#for row_array in data:
#    print(row_array)

correlations = [] #The array for correlations Kij i,j not zero
errors = [] #Errors for the correlations

for meas in data: #xx, xy etc no x0 here
    T = np.sum(meas)
    K = (meas[1]-meas[2]-meas[3]+meas[4])/T
    correlations.append(K)
    s = (1/T) * np.sqrt (meas[1]**2 * ((1/meas[1]) + (1/T)) + meas[2]**2 * ((1/meas[2]) + (1/T)) + meas[3]**2 * ((1/meas[3]) + (1/T)) + meas[4]**2 * ((1/meas[4]) + (1/T)))
    errors.append(s)

c_name = ["Kxx", "Kxy", "Kxz", "Kyx", "Kyy", "Kyz", "Kzx", "Kzy", "Kzz"]
print("correlations no zero")
for name, k, e in zip(c_name, correlations, errors):
    print(f"{name}: {k:.2f} ± {e:.2f}")
    


data_list = [] # collecting the data for i0 coincidences
for u in r_z:
    d = []  
    for index, row in u.iterrows():
        row_values = row.values.tolist()
        d.append(row_values)
    data_list.append(d)  

data_l_list = [] # collecting the data for 0i coincidences   
for u in l_z:
    d = []   
    for index, row in u.iterrows():
        row_values = row.values.tolist()
        d.append(row_values)
    data_l_list.append(d)
    
Cors = [] # the array for correlations Ki0
Ers = [] # errors for these correlations
for t in data_list:
    a1 = [] # collect HH coincidences
    a2 = [] # collect HV coincidences
    a3 = [] # VH
    a4 = [] # VV
    for x in t:
        a1.append(x[1])
        a2.append(x[2])
        a3.append(x[3])
        a4.append(x[4])
    zz = np.sum(a1) # Total HH coincidences
    zo = np.sum(a2) # Total HV coincidences
    oz = np.sum(a3) # Total VH coincidences
    oo = np.sum(a4) # Total VV coincidences
    T = zz + zo + oz + oo # Total relevant coincidences
    K = (zz + zo - oz - oo)/T # Correlations
    szz = np.sqrt(a1[0]**2 + a1[1]**2 + a1[2]**2) # Error for HH coincidences
    szo = np.sqrt(a2[0]**2 + a2[1]**2 + a2[2]**2) # for HV
    soz = np.sqrt(a3[0]**2 + a3[1]**2 + a3[2]**2) # for VH
    soo = np.sqrt(a4[0]**2 + a4[1]**2 + a4[2]**2) # for VV
    sT = np.sqrt(szz**2 + szo**2 + soz**2 + soo**2) # Error for total number of coincidences
    szzT = (zz/T) * np.sqrt((szz/zz)**2 + (sT/T)**2) # error for C_HH/T
    szoT = (zo/T) * np.sqrt((szo/zz)**2 + (sT/T)**2) # for C_HV/T
    sozT = (oz/T) * np.sqrt((soz/zz)**2 + (sT/T)**2) # for C_VH/T
    sooT = (oo/T) * np.sqrt((soo/zz)**2 + (sT/T)**2) # for C_VV/T
    s = np.sqrt(szzT**2 + szoT**2 + sozT**2 + sooT**2) # Final error for the correlations
    Cors.append(K)
    Ers.append(s)
    
cors_name = ["Kx0", "Ky0", "Kz0"]

print("Correlations i0")
for name, k, e in zip(cors_name, Cors, Ers):
    print(f"{name}: {k:.2f} ± {e:.2f}" )


Cors_l = [] # the array for correlations K_0i
Ers_l = [] # the errors for these correlations
for t in data_l_list:
    a1 = [] # collect HH coincidences
    a2 = [] # collect HV coincidences
    a3 = [] # VH
    a4 = [] # VV
    for x in t:
        a1.append(x[1])
        a2.append(x[2])
        a3.append(x[3])
        a4.append(x[4])
    zz = np.sum(a1) # Total HH coincidences
    zo = np.sum(a2) # Total HV coincidences
    oz = np.sum(a3) # Total VH coincidences
    oo = np.sum(a4) # Total VV coincidences
    T = zz + zo + oz + oo # Total relevant coincidences
    K = (zz - zo + oz - oo)/T # Correlations
    szz = np.sqrt(a1[0]**2 + a1[1]**2 + a1[2]**2) # Error for HH coincidences
    szo = np.sqrt(a2[0]**2 + a2[1]**2 + a2[2]**2) # for HV
    soz = np.sqrt(a3[0]**2 + a3[1]**2 + a3[2]**2) # for VH
    soo = np.sqrt(a4[0]**2 + a4[1]**2 + a4[2]**2) # for VV
    sT = np.sqrt(szz**2 + szo**2 + soz**2 + soo**2) # Error for total number of coincidences
    szzT = (zz/T) * np.sqrt((szz/zz)**2 + (sT/T)**2) # error for C_HH/T
    szoT = (zo/T) * np.sqrt((szo/zz)**2 + (sT/T)**2) # for C_HV/T
    sozT = (oz/T) * np.sqrt((soz/zz)**2 + (sT/T)**2) # for C_VH/T
    sooT = (oo/T) * np.sqrt((soo/zz)**2 + (sT/T)**2) # for C_VV/T
    s = np.sqrt(szzT**2 + szoT**2 + sozT**2 + sooT**2) # Final error for the correlations
    Cors_l.append(K)
    Ers_l.append(s)
    
cors_name_l = ["K0x", "K0y", "K0z"]

print("Correlations 0i")
for name, k, e in zip(cors_name_l, Cors_l, Ers_l):
    print(f"{name}: {k:.2f} ± {e:.2f}" )
    
    
idd = np.array([[1, 0], #identity matrix
               [0, 1]])

sx = np.array([[0, 1], # sigma_x
              [1, 0]])

sy = np.array([[0, -1j], # sigma_y
              [1j, 0]])

sz = np.array([[1, 0], # sigma_z
              [0, -1]])

idid = np.kron(idd, idd) #id x id ; x = tensor product

sxid = np.kron(sx, idd) # sigma_x x id
syid = np.kron(sy, idd) # sigma_y x id 
szid = np.kron(sz, idd) # sigma_z x id

idsx = np.kron(idd, sx) # id x sigma_x
idsy = np.kron(idd, sy) # id x sigma_y
idsz = np.kron(idd, sz) # id x sigma_z

sxsx = np.kron(sx, sx) # sigma_x x sigma_x
sxsy = np.kron(sx, sy) # sigma_x x sigma_y
sxsz = np.kron(sx, sz) # sigma_x x sigma_z

sysx = np.kron(sy, sx) # sigma_y x sigma_x , y , z
sysy = np.kron(sy, sy)
sysz = np.kron(sy, sz)

szsx = np.kron(sz, sx) # sigma_z x sigma_x , y, z
szsy = np.kron(sz, sy)
szsz = np.kron(sz, sz)

# Generating the density matrix
ko = Cors[0] * sxid + Cors[1] * syid + Cors[2] * szid # sum of K_i0 terms
ok = Cors_l[0] * idsx + Cors_l[1] * idsy + Cors_l[2] * idsz # sum of K_0i terms
kx = correlations[0]* sxsx + correlations[1] * sxsy  + correlations[2] * sxsz # sum of Kxi terms
ky = correlations[3] * sysx + correlations[4] * sysy + correlations[5] * sysz # sum of Kyi terms
kz = correlations[6] * szsx + correlations[7] * szsy + correlations[8] * szsz # sum of Kzi terms

rhoex = (1/4) * (idid + ko + ok + kx + ky + kz) # measured density matrix

#trace of rhoex
tr = np.trace(rhoex)
print(f"Trace of rhoex: {tr:.2f}")

#eigenvalues of rhoex
rhoex_eigval = np.linalg.eigvals(rhoex)
print("Eigenvalues of rhoex:")
for eiv in rhoex_eigval:
    print(f"{eiv:.2f}")
sum_eigval = np.sum(rhoex_eigval)
print(f"Some of eigenvalues of rhoes: {sum_eigval:.2f}")
    

#fidelity 
rho = np.array([[0.5, 0, 0, -0.5],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-0.5, 0, 0, 0.5]])
F = np.trace(rhoex @ rho)
print(f"Fidelity: {F:.2f}")

#witness
w = np.array([[1, 0, 0, 0],
              [0, 0, -1, 0],
              [0, -1, 0, 0],
              [0, 0, 0, 1]])
W = (1/2) * w
ent = np.trace(W @ rhoex)
print(f"Tr(W * rhoex) = {ent:.2f}")

#peres
rhoexx = rhoex.reshape(2, 2, 2, 2)
rhoexxpt = rhoexx.swapaxes(0, 2)
rhoex_pt = rhoexxpt.reshape(4, 4)
pt_eigval = np.linalg.eigvals(rhoex_pt)
print("Eigenvalues of rhoex_pt:")
for eig in pt_eigval:
    print(f"{eig:.2f}")

#Monte Carlo:
#errors for the matrix elements of rhoex
e00 = 0.25 * np.sqrt(Ers[2]**2 + Ers_l[2]**2 + errors[8]**2)
e01 = 0.25 * np.sqrt(Ers_l[0]**2 + Ers_l[1]**2 + errors[6]**2 + errors[7]**2)
e02 = 0.25 * np.sqrt(Ers[0]**2 + Ers[1]**2 + errors[2]**2 + errors[5]**2)
e03 = 0.25 * np.sqrt(errors[0]**2 + errors[1]**2 + errors[3]**2 + errors[4]**2)
e10 = 0.25 * np.sqrt(Ers_l[0]**2 + Ers_l[1]**2 + errors[6]**2 + errors[7]**2)
e11 = 0.25 * np.sqrt(Ers[2]**2 + Ers_l[2]**2 + errors[8]**2)
e12 = 0.25 * np.sqrt(errors[0]**2 + errors[1]**2 + errors[3]**2 + errors[4]**2)
e13 = 0.25 * np.sqrt(Ers[0]**2 + Ers[1]**2 + errors[2]**2 + + errors[5]**2)
e20 = 0.25 * np.sqrt(Ers[0]**2 + Ers[1]**2 + errors[2]**2 + errors[5]**2)
e21 = 0.25 * np.sqrt(errors[0]**2 + errors[1]**2 + errors[3]**2 + errors[4]**2)
e22 = 0.25 * np.sqrt(Ers[2]**2 + Ers_l[2]**2 + errors[8]**2)
e23 = 0.25 * np.sqrt(Ers_l[0]**2 + Ers_l[1]**2 + errors[6]**2 + errors[7]**2)
e30 = 0.25 * np.sqrt(errors[0]**2 + errors[1]**2 + errors[3]**2 + errors[4]**2)
e31 = 0.25 * np.sqrt(Ers[0]**2 + Ers[1]**2 + errors[2]**2 + errors[5]**2)
e32 = 0.25 * np.sqrt(Ers_l[0]**2 + Ers_l[1]**2 + errors[6]**2 + errors[7]**2)
e33 = 0.25 * np.sqrt(Ers[2]**2 + Ers_l[2]**2 + errors[8]**2)

#errors as matrix, absolute values
E = np.array([[np.abs(e00), np.abs(e01), np.abs(e02), np.abs(e03)],
              [np.abs(e10), np.abs(e11), np.abs(e12), np.abs(e13)],
              [np.abs(e20), np.abs(e21), np.abs(e22), np.abs(e23)],
              [np.abs(e30), np.abs(e31), np.abs(e32), np.abs(e33)]])



# Monte Carlo samples
N_samples = 10000
eigenvalues_samples = np.zeros((N_samples, 4))

# sampling and eigenvalues of samples
for s in range(N_samples):
    rhoex_sample = rhoex + np.random.normal(0, E)
    eigenvalues_samples[s, :] = np.linalg.eigvals(rhoex_sample)


mean_eigenvalues = np.mean(eigenvalues_samples, axis=0) #mean of eigenvalues
std_eigenvalues = np.std(eigenvalues_samples, axis=0) # standard dev of eigenvalues
sem_eigenvalues = std_eigenvalues / np.sqrt(N_samples) # standard error of the mean for eigenvalues

# Print results
for i, (mean, std, sem) in enumerate(zip(mean_eigenvalues, std_eigenvalues, sem_eigenvalues)):
    print(f"Eigenvalue {i+1}:")
    print(f"  Mean: {mean:.2f}")
    print(f"  Standard Deviation: {std:.2f}")
    print(f"  Standard Error of the Mean: {sem:.2f}")


#negativity
negs = [x for x in pt_eigval if x<0]
negativity = np.abs(np.sum(negs))
print(f"negativity: {negativity:.2f}")
dlimit = (2/3) * (1 - 2 * negativity)
print(f"d_min(rhoex) ≥ {dlimit:.2f}")



rhox = rho.reshape(2, 2, 2, 2)
rhopt = rhox.swapaxes(0, 2)
rho_pt = rhopt.reshape(4, 4)
rho_pt_eigval = np.linalg.eigvals(rho_pt)
print("Eigenvalues of rho_pt:")
for eig in rho_pt_eigval:
    print(f"{eig:.2f}")

rho_negs = [y for y in rho_pt_eigval if y<0]
rho_negativity = np.abs(np.sum(rho_negs))
rho_dlimit = (2/3) * (1 - 2 * rho_negativity)
print(f"rho_negativity: {rho_negativity:.2f}")
print(f"d_min(rho) ≥ {rho_dlimit:.2f}")

#logarithmic negativity
rhoex_pt_tr_norm = 1 + 2 * negativity
log_neg = np.log2(rhoex_pt_tr_norm)
print(f"logarithmic negativity: {log_neg:.2f}") 

rho_pt_tr_norm = 1 + 2 * rho_negativity
rho_log_neg = np.log2(rho_pt_tr_norm)
print(f"rhoex_logarithmic negativity: {rho_log_neg:.2f}")

sing = 2 * (1 - ((1 + 2 * negativity)/2))
sing_rho = 2 * (1 - ((1 + 2 * rho_negativity)/2))
print(f"singlet distance ≥ {sing:.2f}")
print(f"singlet distance for bell ≥ {sing_rho:.2f}")

#noisy singlet probability for bell state
p = 1 - (4/3) * ( 1 - (1 + 2 * negativity)/2)
print(f"noisy singlet p = {p:.2f}")
q = 1-p
print(f"noisy singlet: {p:.2f}rho + {q:.2f} (I/4)")

#distinguishing probability
Pr = (1/2) * ( 1 + (1/2 * dlimit))
print(f"Pr ≥ {Pr:.2f}")

#optimum fidelity
Fopt = (1 + 2 * negativity)/2
print(f"F_opt ≤ {Fopt:.2f}")