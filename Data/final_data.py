import requests
from bs4 import BeautifulSoup
from PyAstronomy import pyasl

def get_data(formula, calc, isatom):
    '''
    Inputs: Formula (str), Calculation (str), IsAtom? (bool)
    Supported Calculations: 'geom', 'vibs', 'dipole', 'quadrupole', 'polcalc', 'exp'
    Output: Parsed Data (str)
    '''
    # Get to Level 2 Webpage
    url1 = 'https://cccbdb.nist.gov/%s1x.asp' % calc
    url2 = 'https://cccbdb.nist.gov/%s2x.asp' % calc

    data = {'formula': formula,
            'submit1': 'Submit'}

    headers = {'Host': 'cccbdb.nist.gov',
            'Connection': 'keep-alive',
            'Content-Length': '26',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache',
            'Origin': 'http://cccbdb.nist.gov',
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.302>',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Referer': url1,
            'Accept-Encoding': 'gzip, deflate',
            'Accept-Language': 'en-CA,en-GB;q=0.8,en-US;q=0.6,en;q=0.4'}
    
    session = requests.Session()
    res = session.post('https://cccbdb.nist.gov/getformx.asp', data=data, headers=headers, allow_redirects=False)
    res2 = session.get(url2)
    soup2 = str(BeautifulSoup(res2.content, 'html.parser')).split('\n')
    out = ''
    an = pyasl.AtomicNo()
    basis_list = ['STO-3G', '6-31G*', '6-31G**', '6-31+G**', '6-311G*', '6-311G**', '6-31G(2df,p)', '6-311+G(3df,2p)'\
                       , '6-311+G(3df,2pd)', 'TZVP', 'cc-pVDZ', 'cc-pVTZ', 'cc-pVQZ', 'aug-cc-pVDZ', 'aug-cc-pVTZ'\
                       , 'aug-cc-pVQZ', 'cc-pV(T+d)Z', 'cc-pCVDZ', 'cc-pCVTZ', 'cc-pCVQZ', 'daug-cc-pVDZ', 'daug-cc-pVTZ'\
                       , 'Sadlej_pVTZ']
    # Quadrupole/PolCalc/Exp can be found for all molecules (including atoms)
    if calc == 'quadrupole':
        # Get to Level 3 Webpage
        url3 = 'https://cccbdb.nist.gov/quadrupole3x.asp?method=8&basis=18'# % get_basis(soup2, basis_list)[0]
        res3 = session.get(url3)
        soup3 = str(BeautifulSoup(res3.content, 'html.parser')).split('\n')
        # Find/return data
        i = 0
        start = True
        for line in soup3:
            if 'class=\"num\"' in line:
                end = ' '
                i += 1
                if i < 10:
                    if i == 3:
                        end = '\n                       '
                    elif i % 9 == 0:
                        end = '];\n'
                    elif i % 3 == 0:
                        end = '\n                       '
                    if start:
                        out += 'mol0{im}.quadrupole = ['
                        start = False
                    out += line.strip('<>=/\" abcdefghijklmnopqrstuvwxyz')+end
        return out
    if calc == 'polcalc':
        # Find/return data
        # basis, i = get_basis(soup2, basis_list)
        # basis = 'polcalc3x.asp?method=8&basis='
        basis = 'method=8&amp;basis=18'
        if basis in ''.join(soup2):
            i = -1
            while basis not in soup2[i]:
                i -= 1
            pol = soup2[i].strip('<>=/\" abcdefghijklmnopqrstuvwxyz')
            for i in range(1, len(pol)):
                if pol[-i] not in '-.0123456789':
                    pol = pol[-i+1:]
                    break
            out += 'mol0{im}.polarizability = ['+pol+'];\n'
        return out
    if calc == 'energy':
        # Find/return data
        basis = 'method=8&amp;basis=18'
        if basis in ''.join(soup2):
            i = -1
            while basis not in soup2[i]:
                i -= 1
            Hfg0K = soup2[i].strip('<>=/\" abcdefghijklmnopqrstuvwxyz')
            for i in range(1, len(Hfg0K)):
                if Hfg0K[-i] not in '-.0123456789':
                    Hfg0K = Hfg0K[-i+1:]
                    break
            out += 'mol0{im}.formationEnthalpy0K = ['+Hfg0K+'];\n'
        return out
    if calc == 'exp':
        # Find/return data for Hfg(0K) and IE
        for i in range(len(soup2)):
            if '>Ionization Energy</t' in soup2[i]:
                noIE = True
                j = 1
                while noIE:
                    if 'class=\"num\"' not in soup2[i+j]:
                        j += 1
                    else:
                        noIE = False
                        IE = soup2[i+j].strip('<>=/\" abcdefghijklmnopqrstuvwxyz')
                        out += 'mol0{im}.ionizationEnergy = ['+IE+'];\n'
        return out
    if calc == 'spin':
        for i in range(len(soup2)):
            if 'is closed shell' in soup2[i]:
                out = 'mol0{im}.spinPolarization = 0;\n'
                break
            else:
                if '<td class=\"num\">' in soup2[i]:
                    num = soup2[i].replace('<td class=\"num\">','').replace('<br/>','').replace('</td>','')
                    num = float(num[:len(num)//2])
                    if 0.5<num<1.0:
                        out = 'mol0{im}.spinPolarization = 1;\n'
                    elif 1.75<num<2.25:
                        out = 'mol0{im}.spinPolarization = 2;\n'
                    elif 3.5<num<4.0:
                        out = 'mol0{im}.spinPolarization = 3;\n'
                    elif 5.75<num<6.25:
                        out = 'mol0{im}.spinPolarization = 4;\n'
                    elif 8.5<num<9.0:
                        out = 'mol0{im}.spinPolarization = 5;\n'
                    else:
                        continue
                        #print('Unrecognized num:', num)
                        #out = 'mol0{im}.spinPolarization = ;\n'
                    break
        return out
    if isatom:
        # Return data for atom
        if calc == 'geom':
            out += 'mol0{im}.za = ['+str(an.getAtomicNo(formula.strip('+-1')))+'];\n'
            out += 'mol0{im}.ra = [0.000 0.000 0.000];\n'
        if calc == 'dipole':
            out += 'mol0{im}.dipole = [0.000 0.000 0.000];\n'
        if calc == 'vibs':
            out += 'mol0{im}.vibrations = [];\n'
        return out
    else:
        if calc == 'geom':
            # Get to Level 3 Webpage
            url3 = 'https://cccbdb.nist.gov/geom3x.asp?method=8&basis=18'
            res3 = session.get(url3)
            soup3 = str(BeautifulSoup(res3.content, 'html.parser')).split('\n')
            # Find/return data for ra and za
            ra = []
            za = []
            for i in range(len(soup3)):
                if '<th>x ' in soup3[i]:
                    break
            nonum = True
            j = 1
            while nonum:
                if 'class=\"num\"' not in soup3[i+j]:
                    j += 1
                else:
                    nonum = False
            notend = True
            k = 1
            while notend:
                if 'class=\"num\"' not in soup3[i+j*k]:
                    notend = False
                else:
                    za.append(str(an.getAtomicNo(soup3[i+j*k-1].strip('/td<>0123456789'))))
                    r = soup3[i+j*k].strip('<>=/\" abcdefghijklmnopqrstuvwxyz')+' '
                    r += soup3[i+j*k+1].strip('<>=/\" abcdefghijklmnopqrstuvwxyz')+' '
                    r += soup3[i+j*k+2].strip('<>=/\" abcdefghijklmnopqrstuvwxyz')
                    ra.append(r)
                    k += 1
            out += 'mol0{im}.za = [%s];\n' % ' '.join(za)
            out += 'mol0{im}.ra = [%s];\n' % '\n               '.join(ra)
            return out
        if calc == 'dipole':
            # Get to Level 3 Webpage
            url3 = 'https://cccbdb.nist.gov/dipole3x.asp?method=8&basis=18'
            res3 = session.get(url3)
            soup3 = str(BeautifulSoup(res3.content, 'html.parser')).split('\n')
            # Find/return data
            for i in range(len(soup3)):
                if '<th>x</th>' in soup3[i]:
                    break
            nodip = True
            j = 1
            while nodip:
                if 'class=\"num\"' not in soup3[i+j]:
                    j += 1
                else:
                    nodip = False
                    dip = soup3[i+j].strip('<>=/\" abcdefghijklmnopqrstuvwxyz')+' '
                    dip += soup3[i+j+1].strip('<>=/\" abcdefghijklmnopqrstuvwxyz')+' '
                    dip += soup3[i+j+2].strip('<>=/\" abcdefghijklmnopqrstuvwxyz')
            out += 'mol0{im}.dipole = [%s];\n' % dip
            return out
        if calc == 'vibs':
            # Get to Level 3 Webpage
            url3 = 'https://cccbdb.nist.gov/vibs3x.asp?method=8&basis=18'
            res3 = session.get(url3)
            soup3 = str(BeautifulSoup(res3.content, 'html.parser')).split('\n')
            # Find/return data
            vibs = []
            for i in range(len(soup3)):
                if 'class=\"sym\"' in soup3[i]:
                    vibs.append(soup3[i+1].strip('<>=/\" abcdefghijklmnopqrstuvwxyz'))
                    if 'Π' in soup3[i] or 'E' in soup3[i]:
                        vibs.append(soup3[i+1].strip('<>=/\" abcdefghijklmnopqrstuvwxyz'))
                    if 'T' in soup3[i]:
                        vibs.append(soup3[i+1].strip('<>=/\" abcdefghijklmnopqrstuvwxyz'))
                        vibs.append(soup3[i+1].strip('<>=/\" abcdefghijklmnopqrstuvwxyz'))
            out = 'mol0{im}.vibrations = ['+' '.join(vibs)+'];\n'
            return out
        
def get_basis(soup2, basis_list):
    # Find most accurate method/basis and index of method/basis
    i = 0
    nobasis = True
    while nobasis:
        i -= 1
        for basis in basis_list:
            if '>'+basis+'<' in soup2[i]:
                nobasis = False
                break
    j = 0
    nodata = True
    while nodata:
        j += 1
        if 'class=\"num\"' in soup2[i-j] and 'href=' in soup2[i-j]:
            basis = soup2[i-j].strip('<td class=\"num\"><a href=\"').replace('amp;', '')
            k = basis.index('\">')
            basis = basis[:k]
            nodata = False
    return basis, i-j
                    
def save_data(formula, outfile, init):
    if init <= 10:
        try:
            out = ''
            i = 0
            isatom = False
            if formula.strip('023456789+-') == formula.strip('+-'):
                for char in formula:
                    if char.isupper():
                        i += 1
                if i < 2:
                    isatom = True
            out += 'im=im+1;\n'+'mol0{im}.name = \''+formula+'\';\n'
            for calc in ['geom', 'dipole', 'quadrupole', 'polcalc', 'vibs', 'energy', 'exp']:
                out += get_data(formula, calc, isatom)
            if '+' in formula:
                out += 'mol0{im}.charge = '+str(formula.count('+'))+';\n'
            if '-' in formula:
                out += 'mol0{im}.charge = -'+str(formula.count('-'))+';\n'
            out += '\n'
            f = open(outfile, 'a')
            if isatom or ('vibrations = []' not in out):
                f.write(out)
            else:
                raise Exception('{} missing vibrations'.format(formula))
            return True
        except:
            init += 1
    print('An error has occurred with molecule '+formula+'\n')
    return False
