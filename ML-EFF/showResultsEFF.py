import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.flatten_util import ravel_pytree

from vibrations import vibrations
from mol2feature import mol2feature
import psEeFF


def plotCorrelations(molEFF, molExp, eExp, wExp, model):

    # With Single Atom Energy
    atom_energies = jnp.array([0.0, -0.50226, -2.914694, -7.492054, -14.672447, \
                               -24.663889, -37.859062, -54.60289, -75.09418, -99.76614, \
                               -128.9663, -162.29074, -200.09775, -242.38979, -289.39676, \
                               -341.28625, -398.13928, -460.17532, -527.56])

    # Without Single Atom Energy
    # atom_energies = jnp.array([0.0, -0.40030938, -2.540257, -0.22085778, -0.96117324, -2.3490138, -4.766742, \
    #                            -8.651527, -14.117482, -20.831747, -31.402018, -0.21249142, -0.7910681, -1.9613779, \
    #                            -3.509399, -5.8452907, -9.226815, -13.974816, -19.952742])

    eEFF = jax.vmap(model)(molEFF)

    wEFF = jax.vmap(vibrations, in_axes=(0,None))(molEFF, model)[0]

    aeExp = (jnp.sum(atom_energies[molExp['za']], axis=1) - eExp)
    aeEFF = (jnp.sum(atom_energies[molExp['za']], axis=1) - eEFF)

    featExp = jax.vmap(mol2feature)(molExp)
    featEFF = jax.vmap(mol2feature)(molEFF)

    plt.plot(jnp.arange(-10,11), jnp.arange(-10,11), 'k')
    plt.plot(aeExp, aeEFF, 'b.')
    plt.xlabel('Exp. Atomization Energy (Hart)')
    plt.ylabel('ML-EFF Atomization Energy (Hart)')
    plt.xlim(-1.,1.1)
    plt.ylim(-1.,1.1)
    plt.grid()

    plt.figure()

    plt.plot(jnp.arange(14), jnp.arange(14), 'k')
    #plt.plot(featExp['ee_same'][:,:,-3], featEFF['ee_same'][:,:,-3], 'g.')
    #plt.plot(featExp['ee_opp'][:,:,-3], featEFF['ee_opp'][:,:,-3], 'g.')
    #plt.plot(featExp['ae'][:,:,-2], featEFF['ae'][:,:,-2], 'r.')
    plt.plot(featExp['aa'][:,:,-1], featEFF['aa'][:,:,-1], 'b.')
    plt.xlabel('Exp. Bond Distance (Bohr)')
    plt.ylabel('ML-EFF Bond Distance (Bohr)')
    plt.xlim(0,8)
    plt.ylim(0,8)
    plt.grid()

    #plt.figure()

    #plt.plot(jnp.arange(4), jnp.arange(4), 'k')
    #plt.plot(featExp['e'][:,:,-1], featEFF['e'][:,:,-1], 'b.')
    #plt.xlabel('Exp. Electron Width (Bohr)')
    #plt.ylabel('ML-EFF Electron Width (Bohr)')
    #plt.xlim(0.5,2.5)
    #plt.ylim(0.5,2.5)
    #plt.grid()

    plt.figure()

    plt.plot(2e-2*jnp.arange(2), 2e-2*jnp.arange(2), 'k')
    for i in range(len(wExp)):
        nw = jnp.count_nonzero(wExp[i,:])
        plt.plot(jnp.sort(wExp[i,:nw]),  jnp.sort(wEFF[:,::-1][i,:nw]), 'b.')
    plt.xlabel('Exp. Vibrational Frequency (Hart)')
    plt.ylabel('ML-EFF Vibrational Frequency (Hart)')
    # plt.xlim(0,2e-2)
    # plt.ylim(0,2e-2)
    plt.grid()

    plt.show()


def plotPotentials(model):

    ds = jnp.arange(0.0,4.1,0.1)

    e = model({'za':jnp.array([]), 'ra':jnp.array([]), \
               'rb':jnp.array([[0.],[0.],[0.]]), \
               'wb':jnp.array([1.]), \
               'sb':jnp.array([1])})

    for i in range(1,19):
        a = model({'za':jnp.array([i]), 'ra':jnp.array([[0.],[0.],[0.]]), \
                   'rb':jnp.array([]), \
                   'wb':jnp.array([]), \
                   'sb':jnp.array([])})
        mols = [{'za':jnp.array([i]), 'ra':jnp.array([[0.],[0.],[0.]]), \
                 'rb':jnp.array([[0.],[0.],[d]]), \
                 'wb':jnp.array([1.]), \
                 'sb':jnp.array([1])} for d in ds]

        plt.plot(ds, jnp.array([model(mol) for mol in mols])-a-e-\
                     jnp.array([psEeFF.Exact_AE(jnp.array([i,d,1.])) for d in ds]), label=str(i))
    plt.grid()
    plt.legend(loc='upper right')
    plt.xlabel(r'$r_{iJ}/w_i$', fontsize=14)
    plt.ylabel(r'$U_{cv}(r_{iJ},w_i) \;\; (Hart)$', fontsize=14)
    plt.xlim(0,4)

    plt.figure()

    mols = [{'za':jnp.array([]), 'ra':jnp.array([]), \
             'rb':jnp.array([[0.,0.],[0.,0.],[0.,d*jnp.sqrt(2)]]), \
             'wb':jnp.array([1.,1.]), \
             'sb':jnp.array([1,1])} for d in ds]

    plt.plot(ds, jnp.array([model(mol) for mol in mols])-2*e-\
                 jnp.array([psEeFF.Exact_EE(jnp.array([d,jnp.sqrt(2),1.])) for d in ds]), 'b')
    plt.grid()
    plt.xlabel(r'$r_{ij}/w_{ij}$', fontsize=14)
    plt.ylabel(r'$U_{uu}(r_{ij},w_i,w_j) \;\; (Hart)$', fontsize=14)
    plt.xlim(0,4)

    plt.figure()

    mols = [{'za':jnp.array([]), 'ra':jnp.array([]), \
             'rb':jnp.array([[0.,0.],[0.,0.],[0.,d*jnp.sqrt(2)]]), \
             'wb':jnp.array([1.,1.]), \
             'sb':jnp.array([1,-1])} for d in ds]

    plt.plot(ds, jnp.array([model(mol) for mol in mols])-2*e-\
                 jnp.array([psEeFF.Exact_EE(jnp.array([d,jnp.sqrt(2),1.])) for d in ds]), 'b')
    plt.grid()
    plt.xlabel(r'$r_{ij}/w_{ij}$', fontsize=14)
    plt.ylabel(r'$U_{ud}(r_{ij},w_i,w_j) \;\; (Hart)$', fontsize=14)
    plt.xlim(0,4)

    plt.show()


def plotMolecule(mol, azim=45, elev=10, ax=None):

    iub = mol['sb'] == 1
    idb = mol['sb'] == -1
    
    if ax==None:
        fig = plt.figure(figsize=(6,6))
        ax = plt.axes(projection='3d')
        ax.azim = azim
        ax.elev = elev
        ax.set_xlim([min(mol['ra'][0,:])-2, max(mol['ra'][0,:])+2])
        ax.set_ylim([min(mol['ra'][1,:])-2, max(mol['ra'][1,:])+2])
        ax.set_zlim([min(mol['ra'][2,:])-2, max(mol['ra'][2,:])+2])
        ax.set_box_aspect([1,1,1])
    
    ax.scatter3D(mol['ra'][0,:], mol['ra'][1,:], mol['ra'][2,:], c='k')
    ax.scatter3D(mol['rb'][0,iub], mol['rb'][1,iub], mol['rb'][2,iub], c='b', s=2000*mol['wb'][iub], alpha=0.3)
    ax.scatter3D(mol['rb'][0,idb], mol['rb'][1,idb], mol['rb'][2,idb], c='r', s=2000*mol['wb'][idb], alpha=0.3)
    

def plotBatch(molBatch, num=1, azim=45, elev=10, ax=None, JaxData=None):
    
    ind = num-1
    
    na = jnp.count_nonzero(molBatch['za'][ind])
    nb = jnp.count_nonzero(molBatch['sb'][ind])

    mol = {}
    
    mol['za'] = molBatch['za'][ind,:na]
    mol['sb'] = molBatch['sb'][ind,:nb]
    
    mol['ra'] = molBatch['ra'][ind,:,:na]
    mol['rb'] = molBatch['rb'][ind,:,:nb]
    mol['wb'] = molBatch['wb'][ind,:nb]
    
    if JaxData != None:
        print('Molecule Name:', JaxData[ind]['name'])
    
    plotMolecule(mol, azim=azim, elev=elev)
    

def plotVibrations(mol, pols, vib=1, frac=0, azim=45, elev=10):

    mol_flat, unravel_fn = ravel_pytree({k:mol[k] for k in ['ra', 'rb',  'wb']})

    diff = jnp.array(pols[:,vib-1])
    mol_flat += frac*diff
    
    diff_mol = unravel_fn(mol_flat)
    diff_mol['sb'] = mol['sb']
    
    fig = plt.figure(figsize=(6,6))
    ax = plt.axes(projection='3d')
    ax.azim = azim
    ax.elev = elev
    ax.set_xlim([min(mol['ra'][0,:])-2, max(mol['ra'][0,:])+2])
    ax.set_ylim([min(mol['ra'][1,:])-2, max(mol['ra'][1,:])+2])
    ax.set_zlim([min(mol['ra'][2,:])-2, max(mol['ra'][2,:])+2])
    ax.set_box_aspect([1,1,1])
    
    plotMolecule(diff_mol, azim=azim, elev=elev, ax=ax)
