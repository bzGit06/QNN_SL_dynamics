import numpy as np
from scipy.stats import unitary_group
import qutip as qt
from opt_einsum import contract
from itertools import combinations

def rstHaarUnitary(d, Ndata, num):
    '''
    restricted Haar unitary ensemble
    true label can only be \pm 1; the number of data in each class is same
    '''
    np.random.seed()
    Us = np.zeros((num, d, d), dtype=np.complex128)
    Us[:, np.arange(Ndata), np.arange(Ndata)] = 1.
    if Ndata < d-1:
        Us[:, Ndata:, Ndata:] = unitary_group.rvs(dim=d-Ndata, size=num)
    elif Ndata == d-1:
        Us[:, -1, -1] = np.exp(1j*np.random.uniform(0, 2*np.pi, size=num))
    return Us

def gagb_rstU(X, Os, psis, a, b, num):
    d = len(X)
    U = rstHaarUnitary(d, len(Os), num)
    U1 = unitary_group.rvs(dim=d, size=num)
    U1_hc = U1.conj().transpose(0, 2, 1)
    U2 = contract('bij, bjk->bik', U, U1_hc)

    Oau = contract('bij, jk, bkl->bil', U2.conj().transpose(0, 2, 1), Os[a], U2)
    Obu = contract('bij, jk, bkl->bil', U2.conj().transpose(0, 2, 1), Os[b], U2)

    Ca = contract('ij, bjk->bik', X, Oau) - contract('bij, jk->bik', Oau, X)
    Cb = contract('ij, bjk->bik', X, Obu) - contract('bij, jk->bik', Obu, X)

    ga = contract('i, bij, bjk, bkl, l->b', psis[a].conj(), U1_hc, Ca, U1, psis[a])
    gb = contract('i, bij, bjk, bkl, l->b', psis[b].conj(), U1_hc, Cb, U1, psis[b])

    return -ga*gb/4

def K_rstU(Os, psis, L, Ndata, num):
    paulis = [qt.qeye(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    pauli_idx = np.random.choice(range(1, 4), size=n)
    X = qt.tensor([paulis[j] for j in pauli_idx]).full()

    K = np.zeros((num, Ndata, Ndata))
    for (i, j) in combinations_with_replacement(range(Ndata), 2):
        K[:, i, j] = L*gagb_rstU(X, Os, psis, i, j, num)
        if i != j:
            K[:, j, i] = K[:, i, j]
    return K

def gahagb_rstU_l(X, Os, psis, a, b, num):
    d = len(X)
    U = rstHaarUnitary(d, len(Os), num)
    U1 = unitary_group.rvs(dim=d, size=num)
    U1_hc = U1.conj().transpose(0, 2, 1)
    U2 = contract('bij, bjk->bik', U, U1_hc)
    
    Oau = contract('bij, jk, bkl->bil', U2.conj().transpose(0, 2, 1), Os[a], U2)
    Obu = contract('bij, jk, bkl->bil', U2.conj().transpose(0, 2, 1), Os[b], U2)

    Ca = contract('ij, bjk->bik', X, Oau) - contract('bij, jk->bik', Oau, X)
    Cb = contract('ij, bjk->bik', X, Obu) - contract('bij, jk->bik', Obu, X)
    CCa = contract('ij, bjk->bik', X, Ca) - contract('bij, jk->bik', Ca, X)

    ga = contract('i, bij, bjk, bkl, l->b', psis[a].conj(), U1_hc, Ca, U1, psis[a])
    gb = contract('i, bij, bjk, bkl, l->b', psis[b].conj(), U1_hc, Cb, U1, psis[b])
    ha = contract('i, bij, bjk, bkl, l->b', psis[a].conj(), U1_hc, CCa, U1, psis[a])

    return ga*ha*gb/16

def gahagb_rstU_l1l2(Xs, Os, psis, a, b, num):
    d = len(Xs[0])
    U = rstHaarUnitary(d, len(Os), num)
    U1 = unitary_group.rvs(dim=d, size=num) # U_{l^-}
    U12 = unitary_group.rvs(dim=d, size=num) # U_{l -> l'}
    
    U1_hc = U1.conj().transpose(0, 2, 1) # U_{l^-}^\dagger
    U1c = contract('bij, bjk->bik', U, U1_hc) # U_{l^+}
    U12_hc = U12.conj().transpose(0, 2, 1) # U_{l -> l'}^\dagger
    U2 = contract('bij, bjk, bkl->bil', U, U1_hc, U12_hc) # U_{l'^+}
    U2c = contract('bij, bjk->bik', U2.conj().transpose(0, 2, 1), U) # U_{l'^-}

    Oa_u1 = contract('bij, jk, bkl->bil', U1c.conj().transpose(0, 2, 1), Os[a], U1c)
    Oa_u2 = contract('bij, jk, bkl->bil', U2.conj().transpose(0, 2, 1), Os[a], U2)
    Ob_u2 = contract('bij, jk, bkl->bil', U2.conj().transpose(0, 2, 1), Os[b], U2)

    Ca_u1 = contract('ij, bjk->bik', Xs[0], Oa_u1) - contract('bij, jk->bik', Oa_u1, Xs[0])
    Cb_u2 = contract('ij, bjk->bik', Xs[1], Ob_u2) - contract('bij, jk->bik', Ob_u2, Xs[1])
    Ca_u2 = contract('ij, bjk->bik', Xs[1], Oa_u2) - contract('bij, jk->bik', Oa_u2, Xs[1])
    Ca_u2_u12 = contract('bij, bjk, bkl->bil', U12_hc, Ca_u2, U12)
    CCa = contract('ij, bjk->bik', Xs[0], Ca_u2_u12) - contract('bij, jk->bik', Ca_u2_u12, Xs[0])

    ga = contract('i, bij, bjk, bkl, l->b', psis[a].conj(), U1_hc, Ca_u1, U1, psis[a])
    ha = contract('i, bij, bjk, bkl, l->b', psis[a].conj(), U1_hc, CCa, U1, psis[a])
    gb = contract('i, bij, bjk, bkl, l->b', psis[b].conj(), U2c.conj().transpose(0, 2, 1), Cb_u2, 
                    U2c, psis[b])

    return ga*ha*gb/16

def muRed_rstU(Os, psis, L, Ndata, num):
    paulis = [qt.qeye(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    pauli_idx = np.random.choice(range(1, 4), size=(ndata, n))
    Xs = np.zeros((ndata, 2**n, 2**n), dtype=complex)
    for i in range(ndata):
        Xs[i] = qt.tensor([paulis[j] for j in pauli_idx[i]]).full()

    mu_reduce = np.zeros((num, Ndata, Ndata))
    for (i, j) in product(range(Ndata), repeat=2):
        each = L*gahagb_rstU_l(Xs[0], Os, psis, i, j, num) \
            + L*(L-1)*gahagb_rstU_l1l2(Xs, Os, psis, i, j, num)
        mu_reduce[:, i, j] = each
    return mu_reduce