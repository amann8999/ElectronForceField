import jax
import numpy as np
import jax.numpy as jnp

from molSet import molName


def mat2jax(matdata, molSet):
    matdata = [item for item in matdata if ('formationEnthalpy0K' in item) and (item['name'] in molName[molSet])]
    jaxdata = [dict() for item in matdata]
    for i in range(len(matdata)):
        if len(matdata[i]) != 0:
            for key in matdata[i].keys():
                if type(matdata[i][key])==np.ndarray:
                    if key == 'ra' or key == 'rb':
                        jaxdata[i][key] = jnp.float64(jnp.reshape(jnp.array(matdata[i][key]), (3,-1)))
                    else:
                        jaxdata[i][key] = jnp.array(matdata[i][key])
                elif type(matdata[i][key])==int or type(matdata[i][key])==float:
                    jaxdata[i][key] = jnp.array([matdata[i][key]])
                else:
                    jaxdata[i][key] = matdata[i][key]
    return jaxdata


def data2mol(data, energy_field='formationEnthalpy0K'):
    max_za = max([len(mol['za']) for mol in data])
    max_sb = max([len(mol['sb']) for mol in data])
    max_nw = max([len(mol['vibrations']) for mol in data])
    za = []; ra = []; rb = []; wb = []; sb = []
    E = []; W = []
    for mol in data:
        E.append(mol[energy_field][0])
        nw = len(mol['vibrations'])
        W.append(jnp.zeros(max_nw).at[:nw].set(mol['vibrations']))
        za.append(jnp.pad(mol['za'], (0, max_za-len(mol['za']))))
        ra.append(jnp.pad(mol['ra'], ((0, 0), (0, max_za-jnp.size(mol['za'])))))
        sb.append(jnp.pad(mol['sb'], (0, max_sb-len(mol['sb']))))
        rb.append(jnp.pad(mol['rb'], ((0, 0), (0, max_sb-jnp.size(mol['wb'])))))
        wb.append(jnp.pad(mol['wb'], (0, max_sb-len(mol['sb']))))
    za = jnp.array(za); ra = jnp.array(ra)
    rb = jnp.array(rb); wb = jnp.array(wb); sb = jnp.array(sb)
    E = jnp.array(E); W = jnp.array(W)
    mol_dict = {'za': za, 'ra': ra, 'rb': rb, 'wb': wb, 'sb': sb}
    return mol_dict, E, W
