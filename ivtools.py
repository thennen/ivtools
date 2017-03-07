'''
Tools for working with IV loops in python.

Loops should be in the format (Varray, Iarray)

Some function operate on sets of IV loops, which can just be a list of loops

What should it do??

smooth, with different options
split (into increasing/decreasing and into loops)
Find switches
Find resistance states
Polynomial fits to regions
find anomalous loops
Interpolate to new I/V values
save in/ load from a sensible format
load from insensible formats

plot many different things
scatter parameters by cycle
plot distributions
IV loops with many options
make movies

Data type for a data set is a dict with keys 'iv', and 'meta'
'Meta' is a dict containing metadata about the whole series
'iv' is a numpy array of iv loop dicts
the iv loop dicts have at least keys 'I', 'V', but can contain other arrays and information

Tyler Hennen 2017
'''

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import heapq
import re
import os
import fnmatch
from dotdict import dotdict
from itertools import groupby
from operator import getitem


###################################################################
# Functions for saving and loading data                          ##
###################################################################

pjoin = os.path.join
splitext = os.path.splitext

def load_from_txt(directory, pattern, **kwargs):
    ''' Load list of loops from separate text files. Specify files by glob
    pattern.  kwargs are passed to loadtxt'''
    fnames = fnmatch.filter(os.listdir(directory), pattern)

    # Try to sort by file number, even if fixed width numbers are not used
    # For now I will assume the filename ends in _(somenumber)
    try:
        fnames.sort(key=lambda fn: int(splitext(fn.split('_')[-1])[0]))
    except:
        print('Failed to sort files by file number')

    print('Loading the following files:')
    print('\n'.join(fnames))

    fpaths = [pjoin(directory, fn) for fn in fnames]

    ### List of np arrays version ..
    # Load all the data
    # loadtxt_args = {'unpack':True,
    #                 'usecols':(0,1),
    #                 'delimiter':'\t',
    #                 'skiprows':1}
    # loadtxt_args.update(kwargs)
    # return [np.loadtxt(fp, **loadtxt_args) for fp in fpaths]

    ### Array of DataFrames version
    readcsv_args = dict(sep='\t', decimal='.')
    readcsv_args.update(kwargs)
    def txt_iter():
        # Iterate through text files, load data, and modify in some way
        # Using pandas here only because its read_csv can handle comma decimals easily..
        # Will convert back to numpy arrays.
        for fp in fpaths:
            # TODO: Guess which column has Voltage and Current based on various
            # different names people give them.  Here it seems the situation
            # is very bad and sometimes there are no delimiters in the header.
            # Even this is not consistent.

            # For now, read the first row and try to make sense of it
            with open(fp, 'r') as f:
                header = f.readline()
                if header == '*********** I(V) ***********\n':
                    skiprows = 8
                    for _ in range(7):
                        header = f.readline()
                else:
                    skiprows = 1
                # Try to split it by the normal delimiter
                splitheader = header.split(readcsv_args['sep'])
                if len(splitheader) > 1:
                    # Probably these are the column names?
                    colnames = splitheader
                else:
                    # The other format that I have seen is like 'col name [unit]'
                    # with a random number of spaces interspersed. Split after ].
                    colnames = re.findall('[^\]]+\]', header)
                    colnames = [c.strip() for c in colnames]

            df = pd.read_csv(fp, skiprows=skiprows, names=colnames, index_col=False, **readcsv_args)

            # These will be recognized as the Voltage and Current columns
            Vnames = ['Voltage Source (V)', 'Voltage [V]']
            Inames = ['Current Probe (A)', 'Current [A]']
            # Rename columns
            dfcols = df.columns
            if 'V' not in dfcols:
                for Vn in Vnames:
                    if Vn in dfcols:
                        df.rename(columns={Vn:'V'}, inplace=True)
            if 'I' not in dfcols:
                for In in Inames:
                    if In in dfcols:
                        df.rename(columns={In:'I'}, inplace=True)
            yield df
    # Have to make an intermediate list?  Hopefully this does not take too much time/memory
    # Probably it is not a lot of data if it came from a csv ....
    # This doesn't work because it tries to cast each dataframe into an array first ...
    #return np.array(list(txt_iter()))
    #return (list(txt_iter()), dict(source_directory=directory), [dict(filepath=fp) for fp in fpaths])
    datalist = []
    for fp, df in zip(fpaths, txt_iter()):
        dd = dotdict(I=np.array(df['I']), V=np.array(df['V']), filepath=fp)
        datalist.append(dd)
    iv = np.array(datalist)

    # regular dict version
    #iv = np.array([{'I':np.array(df['I']), 'V':np.array(df['V']), 'filepath':fp} for fp, df in zip(fpaths, txt_iter())])
    return dotdict(iv=iv, source_dir=directory)

#**************************************
#**** Not modified yet for new datatype
#**************************************

def write_data(data, fn):
    ''' Write list of [Vin, Vout] to disk '''
    # TODO: switch to less retarded format
    foldpath = datafolderpath()
    np.savez(os.path.join(foldpath, fn), *data)


def write_fig(fn='plot', fig=None):
    ''' Write a figure to disk '''
    foldpath = datafolderpath()
    fpath = os.path.join(foldpath, fn)

    if fig is None:
        fig = plt.gcf()

    fig.savefig(fpath, dpi=200)

    print('Saved to' + fpath)


def write_figs(data, fn=None, folder=None, subfolder=True):
    ''' Save figs for all sweeps to folder.
    fn : filename for data, also subfolder name if subfolder is True.
    folder : folder to save data.  Determine from global variables if None
    '''
    # TODO: don't silently overwrite
    if fn is not None:
        fnprefix = '{}_sweep_'.format(fn)
    else:
        fn = 'noname'
        fnprefix = 'sweep_'

    if subfolder:
        foldpath = datafolderpath(subfolder=fn)
    else:
        foldpath = datafolderpath()

    if not os.path.isdir(foldpath):
        os.makedirs(foldpath)

    wasinteractive = plt.isinteractive()
    plt.ioff()
    for i, d in enumerate(data):
        fig, ax = plt.subplots()
        ax.set_title(str(i))
        plot_data(d, ax=ax, maxsamples=10000)
        fpath = os.path.join(foldpath, '{}{}.png'.format(fnprefix, i))
        fig.savefig(fpath, dpi=200)
        plt.close(fig)
        print('Saved ' + fpath)
    if wasinteractive:
        plt.ion()


def load_data(fn):
    ''' Get the array of [Vin, Vout]s back from file '''
    d = np.load(fn)
    keys = d.keys()
    n = [int(k[4:]) for k in keys]
    d_out = [d['arr_' + str(i)] for i in range(max(n))]

    return d_out



###################################################################
# Functions for data analysis                                    ##
###################################################################

from functools import wraps

def ivfunc(func):
    '''
    Decorator which allows the same function to be used on a single loop, as
    well as a container of loops.

    Decorated function should take a single loop and return anything

    Then this function will also take multiple loops, and return an array of the outputs
    '''
    @wraps(func)
    def func_wrapper(data, *args, **kwargs):
        dtype = type(data)
        if dtype == np.ndarray:
            # Assuming it's an ndarray of iv dicts
            return np.array([func(d, *args, **kwargs) for d in data])
        elif dtype == dotdict:
            return(func(data, *args, **kwargs))
        else:
            print('ivfunc did not understand the input datatype {}'.format(dtype))
    return func_wrapper


@ivfunc
def moving_avg(data, window=5):
    ''' Smooth data with moving avg '''
    V = data['V']
    I = data['I']
    lenV = len(V)
    lenI = len(I)
    if lenI != lenV:
        print('I and V arrays have different length!')
        return data
    if lenI == 0:
        return data
    weights = np.repeat(1.0, window)/window
    smoothV = np.convolve(V, weights, 'valid')
    smoothI = np.convolve(I, weights, 'valid')

    new_data = data.copy()
    new_data.update({'I':smoothI, 'V':smoothV})
    return new_data


@ivfunc
def index_iv(data, index_function):
    '''
    Index all the data arrays inside an iv loop container at once.
    Condition specified by index function, which should take an iv dict and return an indexing array
    '''
    # Determine the arrays that will be split
    # We will select them now just based on which values are arrays with same size as I and V
    lenI = len(data['I'])
    splitkeys = [k for k,v in data.items() if (type(v) == np.ndarray and len(v) == lenI)]
    dataout = data.copy()
    for sk in splitkeys:
        # Apply the filter to all the relevant items
        index = np.array(index_function(data))
        dataout[sk] = dataout[sk][index]
    return dataout

# Don't know about the name of this one yet
@ivfunc
def apply(data, func, column):
    '''
    This applies func to one column of the ivloop, and leaves the rest the same.
    func should take an array and return an array of the same size
    '''
    dataout = data.copy()
    dataout[column] = func(dataout[column])
    return dataout

@ivfunc
def dV_sign(iv):
    '''
    Return boolean array indicating if V is increasing or constant.
    Will not handle noisy data.  Have to dig up the code that I wrote to do that.
    '''
    direction = np.sign(np.diff(iv['V'])) > 0
    # Need the same size array as started with. Categorize the first point same as second
    return np.append(direction[0], direction)

@ivfunc
def interpolate(data, interpvalues, column='I'):
    '''
    Interpolate all the arrays in ivloop to new values of one of the columns
    Right now this sorts the arrays according to "column"
    would be nice if newvalues could be a function, or an array of arrays ...
    '''
    lenI = len(data[column])
    interpkeys = [k for k,v in data.items() if (type(v) == np.ndarray and len(v) == lenI)]
    interpkeys = [ik for ik in interpkeys if ik != column]

    # Get the largest monotonic subsequence of data, with 'column' increasing
    dataout = largest_monotonic(data)

    # not doing this anymore, but might want the code for something else
    #saturated = abs(dataout[column]/dataout[column][-1]) - 1 < 0.0001
    #lastindex = np.where(saturated)[0][0]
    #dataout[column] = dataout[column][:lastindex]

    for ik in interpkeys:
        dataout[ik] = np.interp(interpvalues, dataout[column], dataout[ik])
    dataout[column] = interpvalues

    return dataout


@ivfunc
def largest_monotonic(data, column='I'):
    ''' returns the segment of the iv loop that is monotonic in 'column', and
    spans the largest range of values.  in output, 'column' will be increasing
    Mainly used for interpolation function.

    Could pass in a function that operates on the segments to determine which one is "largest"
    '''
    lenI = len(data[column])
    keys = [k for k,v in data.items() if (type(v) == np.ndarray and len(v) == lenI)]

    # interp has a problem if the function is not monotonically increasing.
    # Find all monotonic sections of the data, use the longest section,
    # reversing it if it's decreasing This will have problems if 'column'
    # contains noisy data.  Deal with this for now by printing some warnings if
    # no segment of significant length is monotonic

    sign = np.sign(np.diff(data[column]))
    # Group by the sign of the first difference to get indices
    gpby = groupby(enumerate(sign, 0), lambda item: sign[item[0]])
    # Making some lists because I can't think of a better way at the moment
    # Sorry for these horrible lines. It's a list of tuples, (direction, (i,n,d,i,c,e,s))
    monolists = [(gp[0], *zip(*list(gp[1]))) for gp in gpby if abs(gp[0]) == 1]
    directions, indices, _ = zip(*monolists)
    segment_endpoints = [(ind[0], ind[-1] + 2) for ind in indices]
    #return segment_endpoints
    #start_indices, end_indices = zip(*[(ind[0], ind[-1] + 1) for ind in indices])
    # Finally, list of (direction, startindex, endindex) for all monotonic segments
    columnsegments = [data[column][start:end] for (start, end) in segment_endpoints]
    segment_spans = [max(vals) - min(vals) for vals in columnsegments]
    largest = np.argmax(segment_spans)
    direction = int(directions[largest])
    startind, endind = segment_endpoints[largest]

    dataout = data.copy()
    for k in keys:
        dataout[k] = dataout[k][startind:endind][::direction]

    return dataout


@ivfunc
def longest_monotonic(data, column='I'):
    ''' returns the largest segment of the iv loop that is monotonic in
    'column'.  in output, 'column' will be increasing Mainly used for
    interpolation function. '''
    lenI = len(data[column])
    keys = [k for k,v in data.items() if (type(v) == np.ndarray and len(v) == lenI)]

    # interp has a problem if the function is not monotonically increasing.
    # Find all monotonic sections of the data, use the longest section,
    # reversing it if it's decreasing This will have problems if 'column'
    # contains noisy data.  Deal with this for now by printing some warnings if
    # no segment of significant length is monotonic

    sign = np.sign(np.diff(data[column]))
    # Group by the sign of the first difference to get indices
    gpby = groupby(enumerate(sign, 0), lambda item: sign[item[0]])
    # Making some lists because I can't think of a better way at the moment
    monolists = [(gp[0], list(gp[1])) for gp in gpby if abs(gp[0]) == 1]
    segment_lengths = [len(gp[1]) for gp in monolists]
    longest = np.argmax(segment_lengths)
    if segment_lengths[longest] < lenI * 0.4:
        print('No monotonic segments longer than 40% of the {} array were found!'.format(column))
    direction = int(monolists[longest][0])
    startind = monolists[longest][1][0][0]
    endind = monolists[longest][1][-1][0] + 2

    dataout = data.copy()
    for k in keys:
        dataout[k] = dataout[k][startind:endind][::direction]

    return dataout


@ivfunc
def normalize(data):
    ''' Normalize by the maximum current '''
    dataout = data.copy()
    maxI = np.max(data['I'])
    dataout['I'] = dataout['I'] / maxI
    return dataout


def plot_iv(data, ax=None, maxsamples=10000, cm='jet', **kwargs):
    '''
    IV loop plotting which can handle single or multiple loops.
    maxsamples : downsample to this number of data points if necessary
    kwargs passed through to ax.plot
    New figure is created if ax=None

    TODO: Plot arbitrary keys on x and y axis.  Make global dictionary for
    finding long names and units
    Maybe pass an arbitrary plotting function
    '''
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_xlabel('Voltage')
    ax.set_ylabel('Current')

    def plot_one_iv(iv, ax, **kwargs):
        l = len(iv['V'])
        if maxsamples is not None and maxsamples < l:
            # Down sample data
            step = int(l/maxsamples)
            V = iv['V'][np.arange(0, l, step)]
            I = iv['I'][np.arange(0, l, step)]
        else:
            V = iv['V']
            I = iv['I']

        return ax.plot(V, I, **kwargs)[0]

    dtype = type(data)
    if dtype == np.ndarray:
        # There are many loops
        line = []
        # Pick colors
        if isinstance(cm, str):
            cmap = plt.cm.get_cmap(cm)
        else:
            cmap = cm
        colors = [cmap(c) for c in np.linspace(0, 1, len(data))]
        for iv, c in zip(data, colors):
            kwargs.update(c=c)
            line.append(plot_one_iv(iv, ax=ax, **kwargs))
    elif dtype == dotdict:
        line = plot_one_iv(data, ax, **kwargs)
    else:
        print('plot_iv did not understand the input datatype {}'.format(dtype))

    return ax, line

@ivfunc
def moving_avg(data, window=5):
    ''' Smooth data with moving avg '''
    V = data['V']
    I = data['I']
    lenV = len(V)
    lenI = len(I)
    if lenI != lenV:
        print('I and V arrays have different length!')
        return data
    if lenI == 0:
        return data
    weights = np.repeat(1.0, window)/window
    smoothV = np.convolve(V, weights, 'valid')
    smoothI = np.convolve(I, weights, 'valid')

    return {'I':I, 'V':V}

#**************************************
#**** Not modified yet for new datatype
#**************************************

def resistance(data, R2=2000., fitrange=None):
    ''' Try to calculate resistance from Vin, Vout curve, given resistance used
    in divider

    fitrange : (-V1, +V2) range to fit data for slope calculation
    '''
    #Vin = np.array(data[1])
    #Vout = np.array(data[0])

    # Assume data is already in np arrays
    Vin = data[1]
    Vout = data[0]
    if fitrange is not None:
        mask = (Vout > fitrange[0]) & (Vout < fitrange[1])
        Vin = Vin[mask]
        Vout = Vout[mask]
    slope = np.polyfit(Vout, Vin/GAIN['B'], 1)[0]
    # For series resistor
    # R = R2 * (1/slope - 1)

    # For compliance circuit.
    R = R2 / slope

    return R


def fast_resistance(data, R2=2000., V=0.1, delta=0.01, origin=(0, 0)):
    ''' Calculate resistance from current values in a narrow range of V '''
    Vin = data[1]
    Vout = data[0]
    y0 = np.mean(Vin[np.abs(Vin) < delta] - origin[0])
    y1 = np.mean(Vin[np.abs(Vin - V) < delta] - origin[1])

    R = R2 * V / (y1 - y0)

    return R


def is_interesting(data):
    ''' Fit a line to the data and return 1 if the fit is bad '''
    pass


def split(data, nloops=None, nsamples=None, fs=None, duration=None, val=None):
    ''' Split data into loops, specifying somehow the length of each loop

    data : (Aarray, Barray)
    '''
    A = data[0]
    B = data[1]
    l = len(A)
    # Calc number of samples each loop has
    if nsamples is None:
        if nloops is None:
            nsamples = fs * duration
        else:
            nsamples = l / float(nloops)

    nloops = int(l / nsamples)

    Asplit = [A[int(i * nsamples):int((i + 1) * nsamples)] for i in range(0, nloops)]
    Bsplit = [B[int(i * nsamples):int((i + 1) * nsamples)] for i in range(0, nloops)]

    return zip(Asplit, Bsplit)


def switching_v(data, level):
    ''' Calculate switching voltage by level detection '''
    up, down = split_updown(data)
    mask = up[0] > 0
    return interp(level, up[1][mask], up[0][mask])


###### From hystools, maybe useful ##########

def easy_params(H, M, satrange=('75%', '95%'), mpercent=10, hpercent=5, plot=False):
    ''' calculate loop parameters by line extrapolation: hc, hn, hs '''
    # TODO: expand fit ranges if they result in poorly conditioned fits
    Ms, offset = normalize(H, M, satrange, fitbranch=True)
    # ignore offset
    offset = 0
    # split loop into two branches, without knowing which is which
    [H1, H2], [M1, M2] = split(H, M)
    maskflag = False
    while maskflag is False:
        Hcmask1 = np.abs(M1) < (mpercent/100. * Ms)
        Hcmask2 = np.abs(M2) < (mpercent/100. * Ms)
        if sum(Hcmask1) > 1 and sum(Hcmask2) > 1:
            maskflag = True
        else:
            mpercent += 1
    Hcfit1 = np.polyfit(H1[Hcmask1], M1[Hcmask1], 1)
    Hcfit2 = np.polyfit(H2[Hcmask2], M2[Hcmask2], 1)
    # Loop should start from max field
    # TODO: generalize for loop starting from any field

    # From first hys branch
    Hc1 = -Hcfit1[1]/Hcfit1[0]
    Hn1 = (Ms+offset-Hcfit1[1])/Hcfit1[0]
    Hs1 = (-Ms-offset-Hcfit1[1])/Hcfit1[0]
    # From second hys branch
    Hc2 = -Hcfit2[1]/Hcfit2[0]
    Hn2 = (-Ms-offset-Hcfit2[1])/Hcfit2[0]
    Hs2 = (Ms+offset-Hcfit2[1])/Hcfit2[0]
    # Calculate averages
    Hc = (np.max([Hc1, Hc2]) - np.min([Hc1, Hc2]))/2
    Hn = (np.max([Hn1, Hn2]) - np.min([Hn1, Hn2]))/2
    Hs = (np.max([Hs1, Hs2]) - np.min([Hs1, Hs2]))/2

    # Fit poly to points near H=0 to find Mr
    maxH = np.max(np.abs(H))
    maskflag = False
    while maskflag is False:
        Mrmask1 = np.abs(H1) < max(H)*hpercent/100
        Mrmask2 = np.abs(H2) < max(H)*hpercent/100
        if sum(Mrmask1) > 2 and sum(Mrmask2) > 2:
            maskflag = True
        else:
            hpercent += 1
    Mrfit1 = np.polyfit(H1[Mrmask1], M1[Mrmask1], 2)
    Mrfit2 = np.polyfit(H2[Mrmask2], M2[Mrmask2], 2)
    Mr1 = np.polyval(Mrfit1, 0)
    Mr2 = np.polyval(Mrfit2, 0)
    Mr = (np.max([Mr1, Mr2]) - np.min([Mr1, Mr2]))/2

    if plot:
        from matplotlib import pyplot as plt
        lenH = len(H)
        maxM = np.max(M)
        minM = np.min(M)
        # plot input hysteresis loop
        plt.plot(H, M)
        plt.hold(True)
        # TODO: be explicit
        # plot saturation lines
        plt.plot(H, [Ms+offset]*len(H))
        plt.plot(H, [-Ms-offset]*len(H))
        # plot Hc lines
        plt.plot(H, np.polyval(Hcfit1, H))
        plt.plot(H, np.polyval(Hcfit2, H))
        # plot Mr mark
        plt.plot([0,0], [Mr1, Mr2], 'o')
        # print parameters on graph

        plt.grid(True)
        plt.ylim((minM*1.1, maxM*1.1))
        plt.xlabel('H')
        plt.ylabel('M')

    pdict = {}
    pdict['Ms'] = Ms
    pdict['Mr'] = Mr
    pdict['Hn'] = Hn
    pdict['Hs'] = Hs
    pdict['Hc'] = Hc
    pdict['Hc1'] = Hc1
    pdict['Hc2'] = Hc2
    pdict['Hn1'] = Hn1
    pdict['Hn2'] = Hn2
    pdict['Hs1'] = Hs1
    pdict['Hs2'] = Hs2
    pdict['Mr1'] = Mr1
    pdict['Mr2'] = Mr2
    pdict['Hcslope1'] = Hcfit1[0]
    pdict['Hcslope2'] = Hcfit2[0]
    pdict['offset'] = offset

    return pdict


def slope(H, M, fitrange=['75%','95%'], method='avg', fitbranch=True):
    ''' calculate slope of loop by fitting a line to a field range '''

    fitrange = valid_fitrange(fitrange, H)

    dHdir = np.diff(H) < 0
    dHdir = np.append(dHdir[0], dHdir)
    # Find range of data for fits
    if fitbranch:
        # Field is high and decreasing
        fitmask1 = dHdir & (H < fitrange[3]) & (H > fitrange[2])
        # Field is low and increasing
        fitmask2 = ~dHdir & (H < fitrange[1]) & (H > fitrange[0])
    else:
        fitmask1 = (H < fitrange[3]) & (H > fitrange[2])
        fitmask2 = (H < fitrange[1]) & (H > fitrange[0])

    fit1 = fit2 = None
    if any(fitmask1):
        fit1 = np.polyfit(H[fitmask1], M[fitmask1], 1)
    if any(fitmask2):
        fit2 = np.polyfit(H[fitmask2], M[fitmask2], 1)

    if method == 'avg':
        if fit1 is not None and fit2 is not None:
            slope = (fit1[0] + fit2[0]) / 2
            offset = (fit1[1] + fit2[1]) / 2
        elif fit1 is not None:
            slope = fit1[0]
            offset = 0
        elif fit2 is not None:
            slope = fit2[0]
            offset = 0
        else:
            raise Exception('No data points in fit range')
    elif method == 'left':
        if fit2 is not None:
            slope = fit2[0]
            offset = 0
        else:
            raise Exception('No data points in fit range')
    elif method == 'right':
        if fit1 is not None:
            slope = fit1[0]
            offset = 0
        else:
            raise Exception('No data points in fit range')

    return slope, offset


def valid_fitrange(fitrange, fieldarray=None):
    '''
    Try to make sense of input fitrange and return a valid fitrange list
    with len 4.  If any elements of fitrange are a percentage, fieldarray must
    be given
    '''
    # Convert any % in fitrange to field values
    fr = []
    if fieldarray is not None:
        maxfield = np.max(fieldarray)
        minfield = np.min(fieldarray)
    for val in fitrange:
        if isinstance(val, str):
            # if this errors, the string wasn't understood or fieldarray wasn't
            # passed.
            fr.append(float(val.strip('% ')) * maxfield/100.)
        else:
            fr.append(val)

    # Convert to len 4
    if len(fr) == 2:
        if fr[0] < 0:
            fr = [fr[0], fr[1], -fr[1], -fr[0]]
        else:
            fr = [-fr[1], -fr[0], fr[0], fr[1]]

    # TODO: make sure fitrange makes sense.
    assert fr[0] < fr[1] < fr[2] < fr[3]

    return fr


def interpM(H, M, newH):
    '''
    Interpolate one hysteresis loop/branch to new H values.  Conscious of
    increasing/decreasing H values. Can use newH with at most one direction
    change.
    '''
    Hsplit, Msplit = split(H, M)
    newHsplit, _ = split(newH, np.zeros(len(newH)))
#    if len(newHsplit) > 2:
#        raise Exception('newH changes direction more than once')
#    if len(Hsplit) > 2:
#        raise Exception('H changes direction more than once')
#
#
#
#    for newHseg in newHsplit:
#        newdir = (newHseg[1] - newHseg[0]) > 0
#        for Hseg in Hsplit:
#            dir = (Hseg[1] - Hseg[0]) > 0
#            if newdir == dir:
#                pass
#
#    return np.append(M1interp, M2interp)


def interpH(H, M, newM):
    ''' Interpolate a hysteresis loop to new M values '''
    pass


def split(H, M, shortest=2, loops=False):
    '''
    Split hysteresis loops into segments by field change direction.  Throw out
    segments shorter than 'shortest'
    '''
    # find indices for direction changes
    H = np.array(H)
    M = np.array(M)
    dHdir = np.diff(H) > 0
    dHindex = np.append([0,0], np.diff(dHdir))
    dHgroup = np.cumsum(dHindex)
    # split between direction changes, throwing out segments if they are
    # too short
    Hsplit = []
    Msplit = []
    for i in range(dHgroup[-1] + 1):
        if len(dHgroup[dHgroup == i]) >= shortest:
            Hsplit.append(H[dHgroup == i])
            Msplit.append(M[dHgroup == i])

    if loops:
        # Put segments together in groups of two to form loops
        Hloops = []
        Mloops = []
        for j in range(0, len(Hsplit), 2):
            try:
                Hloops.append(np.append(Hsplit[j], Hsplit[j+1]))
                Mloops.append(np.append(Msplit[j], Msplit[j+1]))
            except:
                # There aren't two more, but append the stray segment
                Hloops.append(Hsplit[j])
                Mloops.append(Msplit[j])
        return Hloops, Mloops

    return Hsplit, Msplit


def smooth(H, M, window=5):
    if len(H) == 0 or len(M) == 0:
        return H, M
    ''' Smooth H and M with rolling average'''
    weights = np.repeat(1.0, window)/window
    smoothH = np.convolve(H, weights, 'valid')
    smoothM = np.convolve(M, weights, 'valid')
    return smoothH, smoothM


########### These operate on a list of (V, I) #####################



def filt_datalist(datalist, window=5):
    ''' Filter a list of data with moving average '''
    fdata = []
    for d in datalist:
        fdata.append(moving_avg(d, window))
    return fdata


def concat(datalist):
    ''' Put split data back together '''
    concatdata = [[],[]]
    for d in datalist:
        concatdata[0].extend(d[0])
        concatdata[1].extend(d[1])

    return concatdata


def hc(H, M, method='avg', mpercent=10):
    ''' calculate Hc '''
    # split into two branches and fit Hc
    amp = np.max(np.abs(M))
    [H1, H2], [M1, M2] = split(H, M)
    mask1 = np.abs(M1) < (mpercent/100. * amp)
    mask2 = np.abs(M2) < (mpercent/100. * amp)
    Hcfit1 = np.polyfit(H1[mask1], M1[mask1], 1)
    Hcfit2 = np.polyfit(H2[mask2], M2[mask2], 1)
    Hc1 = -Hcfit1[1]/Hcfit1[0]
    Hc2 = -Hcfit2[1]/Hcfit2[0]

    if method == 'avg':
        return np.abs(Hc2 - Hc1) / 2.
    elif method == 'left':
        return np.min((Hc1, Hc2))
    elif method == 'right':
        return np.max((Hc1, Hc2))
    else:
        raise Exception('"method" can be \'left\', \'right\', or \'avg\'')


def recoil(Hlist, Mlist, startdef=0, satrange=('85%', '95%'), plot=False):
    '''
    Extract parameters from recoil data.  Input list of recoil curves, first
    being the major loop.
    startdef:
    0: minor loops start from remnant value
    1: minor loops start from initial value
    '''

    # only doing +- 0.5 point, can be extended later
    Mlevels = [-0.5, 0.5]

    # Find amp and offs from major loop
    amp, offs = normalize(Hlist[0], Mlist[0], fitrange=satrange, fitbranch=False)
    Minterp = np.array([amp*m + offs for m in Mlevels])
    # H interp will be a list of H values corresponding to the M values.  Each
    # list element corresponding to a loop
    Hinterp = []

    Mr = []
    dH_ext = []
    # initial values of H, M for each minor loop
    Minit= []
    Hinit = []
    # Extract remnant values, initial values, dH values
    for H, M in zip(Hlist, Mlist):
        # find 10 points nearest 0, fit a parabola to find intercept.
        argmin = np.abs(H).argmin()
        slice_ = slice(argmin-5, argmin+5)
        pfit = np.polyfit(H[slice_], M[slice_], 2)
        mr = np.polyval(pfit, 0)
        Mr.append(mr)
        # average first 10 points for starting M
        minit= np.mean(M[:10])
        Minit.append(minit)
        # Find M value where minor loop has half-returned to saturation
        if startdef == 0:
            mhalf = mr + (amp + offs - mr)/2
        elif startdef == 1:
            mhalf = minit + (amp + offs - minit)/2

        # interp for Minterp -- could behave badly in regions where M is not
        # monotonic
        h = np.interp(Minterp, M, H, left=np.NaN, right = np.NaN)
        Hinterp.append(h)

        # Calculate dH_ext by subtracting H_minorloop(mhalf) from H_majorloop(-mhalf)
        hhalf = np.interp(mhalf, M, H)
        majhhalf = np.interp(-mhalf+2*offs, Mlist[0], Hlist[0])

        dH_ext.append(hhalf - majhhalf)

    # Calculate SFD parameters
    Hnhalf_major = Hinterp[0][Mlevels.index(-0.5)]
    Hhalf_major = Hinterp[0][Mlevels.index(0.5)]
    dH_int = [Hhalf_major - h[Mlevels.index(0.5)] for h in Hinterp]
    SFD = Hhalf_major - Hnhalf_major

    # Interpolate dH_int, dH_ext, to get iSFD, eSFD
    # First sort everything by Mstart value
    if startdef == 0:
        Mstart, dH_int, dH_ext, Hlist, Mlist = zip(*sorted(zip(Mr, dH_int, dH_ext, Hlist, Mlist)))
    elif startdef == 1:
        Mstart, dH_int, dH_ext, Hlist, Mlist = zip(*sorted(zip(Minit, dH_int, dH_ext, Hlist, Mlist)))
    if max(Mstart) > 0 and min(Mstart) < 0:
        iSFD = np.interp(0, Mstart, dH_int)
        eSFD = np.interp(0, Mstart, dH_ext)
        # interpolate to get entire M=0 loop
        nearz = heapq.nsmallest(2, Mstart, key=abs)
        nearz.sort()
        i_neg = Mstart.index(nearz[0])
        i_pos = Mstart.index(nearz[1])
        interplower = np.interp(Hlist[i_pos], Hlist[i_neg], Mlist[i_neg])
        H_zminor = Hlist[i_pos]
        weight = 1/(1 - nearz[0]/nearz[1])
        #M_zminor = [((1 - weight)*mp + weight*mn) for mp,mn in zip(Mlist[i_pos], interplower)]
        M_zminor = (1-weight)*Mlist[i_pos] + weight*interplower
    else:
        # if minor loop starting points to not cross M=0, fit parabola to dH vs
        # Mstart to get Mstart=0 value
        iSFD_fit = np.polyfit(Mstart, dH_int, 2)
        eSFD_fit = np.polyfit(Mstart, dH_ext, 2)
        iSFD = np.polyval(iSFD_fit, 0)
        eSFD = np.polyval(eSFD_fit, 0)
        # TODO: extrap for M=0 loop
        # for now, using closest to M=0 loop
        nearestz = min(Mstart, key=abs)
        H_zminor = Hlist[Mstart.index(nearestz)]
        M_zminor = Mlist[Mstart.index(nearestz)]


    if plot:
        from matplotlib import pyplot as plt

        # TODO: be explicit
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hold(True)
        for H, M, hinterp in zip(Hlist, Mlist, Hinterp):
            plt.plot(H, M)
            # plot Hinterp, Minterp
            plt.plot(hinterp, Minterp, 'o')
            # print parameters on graph
        try:
            # plot interpolated minor loop
            plt.plot(H_zminor, M_zminor, linewidth=3, color='black')
        except:
            pass

        plt.grid(True)
        plt.xlabel('H')
        plt.ylabel('M')

    pdict = {}
    pdict['SFD'] = SFD
    pdict['iSFD'] = iSFD
    pdict['eSFD'] = eSFD
    pdict['Mstart'] = Mstart
    pdict['H_zminor'] = H_zminor
    pdict['M_zminor'] = M_zminor
    pdict['plotpts'] = [(Hhalf_major, offs+amp/2),
                        (Hnhalf_major, offs-amp/2),
                        (Hhalf_major-iSFD, offs+amp/2)]

    return pdict


###################################################################
# Functions for plotting                                         ##
###################################################################

def plot_resist_cycle(datalist, fitrange=(-.5, .5), alpha=.6):
    ''' Plot resistance of increasing and decreasing loop branches
    fitrange : (-V1, +V2) range to fit data for slope calculation
               (-V1, +V2, -V3, +V4) uses first pair for increasing curves,
                                    second pair for decreasing curves
    '''
    if len(fitrange) == 2:
        fitrange2 = fitrange
    elif len(fitrange) == 4:
        fitrange2 = fitrange[2:]
        fitrange = fitrange[:2]

    resist_up = []
    resist_down = []
    cycle = []
    for n, d in enumerate(datalist):
        try:
            dup, ddown = split_updown(d)
            Rup = resistance(dup, fitrange=fitrange)
            Rdown = resistance(ddown, fitrange=fitrange2)
            resist_up.append(Rup)
            resist_down.append(Rdown)
            cycle.append(n)
        except:
            print('failed to find R on cycle {}'.format(n))

    fig, ax = plt.subplots()
    #ax.plot(resist_up, '.')
    #ax.plot(resist_down, '.')
    ax.scatter(cycle, resist_up, s=10, alpha=alpha, edgecolor='none', c='royalblue')
    ax.scatter(cycle, resist_down, s=10, alpha=alpha, edgecolor='none', c='seagreen')
    ax.legend(['HRS', 'LRS'], loc=0)
    ax.set_xlabel('Cycle #')
    ax.set_ylabel('Resistance / $\\Omega$')

    return resist_up, resist_down


def plot_histogram2d(data, bins=200, logscale=True, cmap='YlGnBu', **kwargs):
    a, b = data
    H, xedges, yedges = np.histogram2d(a, b, bins=bins)
    H = H.transpose()

    Hmasked = np.ma.masked_where(H == 0, H)
    if logscale:
        Hmasked = np.log10(Hmasked)

    fig, ax = plt.subplots()
    im = ax.pcolormesh(xedges, yedges, Hmasked, cmap=cmap, **kwargs)
    ax.set_xlabel('V (Channel A)')
    ax.set_ylabel('V (Channel B)')
    cbar = fig.colorbar(im)
    if logscale:
        cbar.ax.set_ylabel('log10(Counts)')
    else:
        cbar.ax.set_ylabel('Counts')

    return H


def plot_reset_current(datalist, R2=2000, offset=0, **kwargs):
    ''' Plot maximum reset current as function of cycle.
    Return list of reset currents'''
    fig, ax = plt.subplots()
    Ireset = [(min(d[1]) + offset)/R2 for d in data]
    ax.plot([1e6 * I for I in Ireset], '.', **kwargs)
    ax.set_xlabel('Cycle #')
    ax.set_ylabel('Reset Current (uA)')

    return Ireset


def plot_switching_v(datalist, level, **kwargs):
    ''' Plot switching voltage as a function of cycle. Return switching voltage list '''
    Vswitch = [switching_v(d, level) for d in datalist]
    fig, ax = plt.subplots()
    ax.plot(Vswitch, '.', **kwargs)
    ax.set_xlabel('Cycle #')
    ax.set_ylabel('Switching Voltage (V)')

    return Vswitch

### This makes a movie from png frames

def frames_to_mp4(directory, base='Loop', outname='out'):
    # Send command to create video with ffmpeg
    cmd = (r'cd "{}" & ffmpeg -framerate 5 -i {}_%03d.png -c:v libx264 '
            '-r 15 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
            '{}.mp4').format(directory, base, outname)
    os.system(cmd)
