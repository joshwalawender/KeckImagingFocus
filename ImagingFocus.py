#!python3

## Import General Tools
from pathlib import Path
import argparse
import re
import logging

import numpy as np
from astropy.io import fits
from astropy import visualization as vis
from astropy import stats
from astropy.table import Table, Column, vstack
from astropy.modeling import models, fitting
import photutils
import sep

from matplotlib import pyplot as plt


##-------------------------------------------------------------------------
## Parse Command Line Arguments
##-------------------------------------------------------------------------
## create a parser object for understanding command-line arguments
p = argparse.ArgumentParser(description='''
''')
## add flags
p.add_argument("-v", "--verbose", dest="verbose",
    default=False, action="store_true",
    help="Be verbose! (default = False)")
## add options
p.add_argument("-n", "--nfiles", dest="nfiles", type=int, default=7,
    help="The number of files. Negaive value starts from the last file in the sequence.")
## add arguments
p.add_argument('files', nargs='*',
               help="The files to analyze.  If a single file is given, the nfiles argument will be used to find the other files.")
args = p.parse_args()


##-------------------------------------------------------------------------
## Create logger object
##-------------------------------------------------------------------------
log = logging.getLogger('ImagingFocus')
log.setLevel(logging.DEBUG)
## Set up console output
LogConsoleHandler = logging.StreamHandler()
if args.verbose is True:
    LogConsoleHandler.setLevel(logging.DEBUG)
else:
    LogConsoleHandler.setLevel(logging.INFO)
LogFormat = logging.Formatter('%(asctime)s %(levelname)8s: %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
LogConsoleHandler.setFormatter(LogFormat)
log.addHandler(LogConsoleHandler)
## Set up file output
# LogFileName = None
# LogFileHandler = logging.FileHandler(LogFileName)
# LogFileHandler.setLevel(logging.DEBUG)
# LogFileHandler.setFormatter(LogFormat)
# log.addHandler(LogFileHandler)


##-------------------------------------------------------------------------
## analyze_hdu
##-------------------------------------------------------------------------
def analyze_hdu(hdu, extract_thresh=5, minarea=4, minflux=10000, plot=True,
                minab=1, maxab=30):
    ## SEP
    log.debug('Subtracting background')
    bkg = photutils.Background2D(hdu.data, box_size=128,
                                 sigma_clip=stats.SigmaClip())
    log.debug('Extracting objects')
    objects = Table(sep.extract(hdu.data-bkg.background,
                                thresh=extract_thresh,
                                minarea=minarea) )
    log.debug('Filtering objects')
    objects = objects[objects['a'] > minab]
    objects = objects[objects['b'] > minab]
    objects = objects[objects['a'] < maxab]
    objects = objects[objects['b'] < maxab]
    objects = objects[objects['flux'] > minflux]

    log.debug('Calculating FWHMs')
    coef = 2*np.sqrt(2*np.log(2))
    fwhm = np.sqrt((coef*objects['a'])**2 + (coef*objects['b'])**2)
    objects.add_column(Column(data=fwhm.data, name='fwhm', dtype=np.float))

    if plot is True:
        log.debug('Generating plot')
        plt.figure(figsize=(12,12))
        norm = vis.ImageNormalize(hdu.data, interval=vis.PercentileInterval(99))
        plt.imshow(hdu.data, origin='lower', cmap='gray', norm=norm)
        for source in objects:
            circle = plt.Circle((source['x'], source['y']), radius=10,
                                 edgecolor='r', facecolor='none')
            plt.gca().add_artist(circle)

    return objects


##-------------------------------------------------------------------------
## analyze_file
##-------------------------------------------------------------------------
def analyze_file(file, extract_thresh=5, minarea=4, minflux=10000, plot=True,
                 focuskeyword='FOCUS', ccdnamekeyword='CCDNAME'):
    hdul = fits.open(file)
    for hdu in hdul:
        focusval = hdu.header.get(focuskeyword, None)
        if focusval is not None:
            break
    hdul = [hdu for hdu in hdul if hdu.data is not None \
            and type(hdu) in [fits.PrimaryHDU, fits.ImageHDU]]
    extracted = {}
    log.info(f"Analyzing {len(hdul)} image HDUs in {file.name} with focus = {focusval:4g}")
    for i,hdu in enumerate(hdul):
        ccdname = hdu.header.get(ccdnamekeyword, None)
        try:
            log.debug(f'  Analyzing HDU {i}')
            objects = analyze_hdu(hdu, extract_thresh=extract_thresh,
                                  minarea=minarea, minflux=minflux, plot=plot)
            if len(objects) > 0:
                log.info(f"  HDU {i} (CCD {ccdname}): found {len(objects)} sources with median FWHM of {np.median(objects['fwhm']):.2f} pix")
            else:
                log.info(f"  HDU {i} (CCD {ccdname}): found {len(objects)} sources")
                objects = None
        except Exception as e:
            objects = None
            log.warning(e)
        if objects is not None:
            if ccdname in extracted.keys():
                extracted[ccdname] = vstack(extracted[ccdname], objects)
            else:
                extracted[ccdname] = objects
    fwhm = [np.median(extracted[ccdname]['fwhm']) for ccdname in extracted.keys()]

    return focusval, fwhm, [key for key in extracted.keys()]


##-------------------------------------------------------------------------
## analyze_focus_run
##-------------------------------------------------------------------------
def analyze_focus_run(files, minflux=10000, extract_thresh=5, minarea=4,
                      focuskeyword='FOCUS', ccdnamekeyword='CCDNAME'):
    focusvals = [analyze_file(file, minflux=minflux, focuskeyword=focuskeyword,
                              ccdnamekeyword=ccdnamekeyword, plot=False)
                              for file in files]
    focusvals = Table(np.array(focusvals), names=('focus', 'fwhm', 'ccdname'))
    
    nfwhms = len(focusvals['fwhm'][0])
    p0 = models.Polynomial1D(degree=2)
    fit = fitting.LinearLSQFitter()

    parabolas = [None]*nfwhms
    bestfocus = [np.nan]*nfwhms

    plt.figure(figsize=(10,10))

    colors = ['b', 'g', 'r', 'k']
    for i in range(nfwhms):
        fwhms = [val['fwhm'][i] for val in focusvals
                 if not np.isnan(val['fwhm'][i])]
        ccdnames = [val['ccdname'][i] for val in focusvals
                    if not np.isnan(val['fwhm'][i])]
        focus = [val['focus'] for val in focusvals
                 if not np.isnan(val['fwhm'][i])]
        if len(focus) > 4:
            p = fit(p0, focus, fwhms)
            bestfocus[i] = -p.c1/2/p.c2
            parabolas[i] = p
            plt.plot(focus, fwhms, f'{colors[i]}o',
                     label=f'CCD {ccdnames[i]}: {bestfocus[i]:.4g}')
            focuspoints = np.linspace(min(focus), max(focus), 100)
            plt.plot(focuspoints, p(focuspoints), f'{colors[i]}-', alpha=0.3)
            plt.axvline(bestfocus[i], color=colors[i], alpha=0.3)
            
    mean_bestfocus = np.nanmean(bestfocus)
    log.info(f"Mean Best Focus = {mean_bestfocus:.4g}")

    plt.legend(loc='best')
    plt.xlabel('Focus Value')
    plt.ylabel('FWHM (pix)')
    plt.title(f'Best Focus = {mean_bestfocus:.4g}')
    plt.show()

    return mean_bestfocus


if __name__ == '__main__':
    if len(args.files) > 1:
        inputfiles = [Path(f) for f in args.files]
    else:
        first_file = Path(args.files[0])
        matchfilename = re.search('(.*[a-zA-Z_\-])(\d+).fits', first_file.name)
        if matchfilename is not None:
            basename = matchfilename.group(1)
            numcount = len(matchfilename.group(2))
            firstnum = int(matchfilename.group(2))
            sign = lambda a: (a>0) - (a<0)
            inputfiles = [first_file.parent / Path(f"{basename}{i:0{numcount}d}.fits")
                          for i in range(firstnum, firstnum+args.nfiles, sign(args.nfiles))]

    header = fits.getheader(inputfiles[0])
    inst = header.get('INSTRUME')
    log.info(f'Found INSTRUME: {inst}')
    if re.match('^LRISBLUE', inst):
        focuskeyword='BLUFOCUS'
    elif re.match('^LRIS$', inst):
        focuskeyword='REDFOCUS'
    elif re.match('^DEIMOS', inst):
        focuskeyword='DWFOCVAL'

    analyze_focus_run(inputfiles, minflux=minflux, minarea=minarea,
                      extract_thresh=extract_thresh, 
                      focuskeyword=focuskeyword, ccdnamekeyword=ccdnamekeyword)

# Example Calls
# python ImagingFocus.py ~/OneDrive\ -\ keck.hawaii.edu/InstrumentFocus/LRIS/2020aug18_B/bfoc000[1-7].fits
# python ImagingFocus.py ~/OneDrive\ -\ keck.hawaii.edu/InstrumentFocus/LRIS/2020aug18_B/bfoc0007.fits --nfiles=-7
# python ImagingFocus.py ~/OneDrive\ -\ keck.hawaii.edu/InstrumentFocus/LRIS/2020aug18_B/rfoc0008.fits --nfiles=7
# python ImagingFocus.py ~/OneDrive\ -\ keck.hawaii.edu/InstrumentFocus/DEIMOS/2020aug20/d0820_0002.fits --nfiles=7