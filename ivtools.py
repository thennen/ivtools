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
first pos jump
last neg jump
threshon
threshoff
fitpoly
plotpoly
calc resistance/differential resistance

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

# TODO: Also save somehow to this retarded format:

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
    for i, (fp, df) in enumerate(zip(fpaths, txt_iter())):
        mtime = os.path.getmtime(fp)
        ctime = os.path.getctime(fp)
        longnames = {'I':'Current', 'V':'Voltage'}
        units = {'I':'A', 'V':'V'}
        dd = dotdict(I=np.array(df['I']), V=np.array(df['V']), filepath=fp,
                     mtime=mtime, ctime=ctime, units=units, longnames=longnames,
                     index=i)
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

    Don't know if this is a good idea or not ...

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

@ivfunc
def slice_iv(data, stop, start=0, step=None):
    '''
    Slice all the data arrays inside an iv loop container at once.
    start, stop can be functions that take the iv loop as argument
    '''
    lenI = len(data['I'])
    splitkeys = [k for k,v in data.items() if (type(v) == np.ndarray and len(v) == lenI)]
    dataout = data.copy()
    if callable(start):
        start = start(data)
    if callable(stop):
            stop = stop(data)
    for sk in splitkeys:
        # Apply the filter to all the relevant items
        dataout[sk] = dataout[sk][slice(start, stop, step)]
    return dataout



# Don't know about the name of this one yet
@ivfunc
def apply(data, func, column):
    '''
    This applies func to one column of the ivloop, and leaves the rest the same.
    func should take an array and return an array of the same size
    TODO:  Do this in place!
    '''
    dataout = data.copy()
    dataout[column] = func(dataout[column])
    return dataout

def insert(data, key, vals):
    # Insert values into ivloop objects
    for d,v in zip(data, vals):
        d[key] = v

def extract(data, key):
    # Get array of values from ivloop objects
    return array([d[key] for d in data])

@ivfunc
def dV_sign(iv):
    '''
    Return boolean array indicating if V is increasing, decreasing, or constant.
    Will not handle noisy data.  Have to dig up the code that I wrote to do that.
    '''
    direction = np.sign(np.diff(iv['V']))
    # Need the same size array as started with. Categorize the last point same as previous 
    return np.append(direction, direction[-1])

@ivfunc
def decreasing(iv):
    return index_iv(iv, lambda l: dV_sign(iv) < 0)


@ivfunc
def increasing(iv):
    return index_iv(iv, lambda l: dV_sign(iv) > 0)


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
def jumps(loop, column='I', thresh=0.25, normalize=True, abs=True):
    ''' Find jumps in the data.
    if normalize=True, give thresh as fraction of maximum absolute value.
    return (indices,), (values of jumps,)
    pass abs=False if you care about the sign of the jump
    '''
    d = diff(loop[column])
    if normalize:
        thresh = thresh * np.max(np.abs(loop[column]))
    # Find jumps greater than thresh * 100% of the maximum
    if abs:
        jumps = np.where(np.abs(d) > thresh )[0]
    elif thresh < 0:
        jumps = np.where(d < thresh)[0]
    else:
        jumps = np.where(d > thresh)[0]
    return jumps, d[jumps]

@ivfunc
def njumps(loop, **kwargs):
    j = jumps(loop, **kwargs)
    njumps = len(j[0])
    loop['njumps'] = njumps
    return njumps


@ivfunc
def first_jump(loop, **kwargs):
    j = jumps(loop, **kwargs)
    if any(j):
        first_jump = j[0][0]
    else:
        first_jump = np.nan
    loop['first_jump'] = first_jump
    return first_jump

@ivfunc
def last_jump(loop, **kwargs):
    j = jumps(loop, **kwargs)
    if any(j):
        last_jump = j[0][-1]
    else:
        last_jump = np.nan
    loop['last_jump'] = last_jump
    return last_jump 


def pindex(loops, column, index):
    # Index some column of all the ivloops in parallel
    # "index" is a list of indices with same len as loops
    # Understands list[nan] --> nan
    # TODO: index by a number contained in the ivloop object
    vals = []
    for l,i in zip(loops, index):
        if isnan(i):
            vals.append(np.nan)
        else:
            vals.append(l[column][int(i)])
    return array(vals)



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


def plot_one_iv(iv, ax=None, x='V', y='I', maxsamples=10000, **kwargs):
    ''' Plot an array vs another array contained in iv object '''
    if ax is None:
        fig, ax = plt.subplots()

    if type(y) == str:
        Y = iv[y]
    else:
        Y = y
    l = len(Y)

    if x is None:
        X = np.arange(l)
    elif type(x) == str:
        X = iv[x]
    else:
        X = x
    if maxsamples is not None and maxsamples < l:
        # Down sample data
        step = int(l/maxsamples)
        X = X[np.arange(0, l, step)]
        Y = Y[np.arange(0, l, step)]

    # Try to name the axes according to metadata
    # Will error right now if you pass array as x or y
    if x == 'V': longnamex = 'Voltage'
    elif x is None:
        longnamex = 'Data Point'
    elif type(x) == str:
        longnamex = x
    if y == 'I': longnamey = 'Current'
    else: longnamey = y
    if 'longnames' in iv.keys():
        if x in iv['longnames'].keys():
            longnamex = iv['longnames'][x]
        if y in iv['longnames'].keys():
            longnamey = iv['longnames'][y]
    if x is None: unitx = '#'
    else: unitx = '?'
    unity = '?'
    if 'units' in iv.keys():
        if x in iv['units'].keys():
            unitx = iv['units'][x]
        if y in iv['units'].keys():
            unity = iv['units'][y]

    ax.set_xlabel('{} [{}]'.format(longnamex, unitx))
    ax.set_ylabel('{} [{}]'.format(longnamey, unity))

    return ax.plot(X, Y, **kwargs)[0]

def plotiv(data, x='V', y='I', ax=None, maxsamples=10000, cm='jet', **kwargs):
    '''
    IV loop plotting which can handle single or multiple loops.
    maxsamples : downsample to this number of data points if necessary
    kwargs passed through to ax.plot
    New figure is created if ax=None

    Maybe pass an arbitrary plotting function
    '''
    if ax is None:
        fig, ax = plt.subplots()

    dtype = type(data)
    if dtype == np.ndarray:
        # There are many loops
        if x is None or hasattr(data[0][x], '__iter__'):
            line = []
            # Pick colors
            if isinstance(cm, str):
                cmap = plt.cm.get_cmap(cm)
            else:
                cmap = cm
            colors = [cmap(c) for c in np.linspace(0, 1, len(data))]
            for iv, c in zip(data, colors):
                kwargs.update(c=c)
                line.append(plot_one_iv(iv, ax=ax, x=x, y=y, maxsamples=maxsamples, **kwargs))
        else:
            # Probably referencing scalar values.
            # No tests to make sure both x and y scalar values for all loops.
            X = extract(data, x)
            Y = extract(data, y)
            line = ax.plot(X, Y, **kwargs)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
    elif dtype == dotdict:
        line = plot_one_iv(data, ax, x=x, y=y, maxsamples=maxsamples, **kwargs)
    else:
        print('plotiv did not understand the input datatype {}'.format(dtype))

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


def concat(datalist):
    ''' Put split data back together '''
    concatdata = [[],[]]
    for d in datalist:
        concatdata[0].extend(d[0])
        concatdata[1].extend(d[1])

    return concatdata


###################################################################
# Functions for plotting                                         ##
###################################################################


def write_pngs(data, directory, plotfunc=plot_one_iv, **kwargs):
    '''
    Write a png of each of the iv loops to disk.  kwargs passed to plot()
    
    TODO: Pass an arbitrary plotting function
    '''
    if not os.path.isdir(directory):
        os.makedirs(directory)
    fig, ax = plt.subplots()
    for i, ivloop in enumerate(data):
        line = plotfunc(ivloop, ax=ax, **kwargs)
        title = 'Loop {}'.format(i)
        if 'filepath' in ivloop.keys():
            fp = ivloop['filepath']
            origin_fname = os.path.split(fp)[1]
            fname = os.path.splitext(origin_fname)[0] + '.png'
            title = '{}: {}'.format(title, origin_fname)
        else:
            fname = 'Loop_{:03d}'.format(i)

        ax.set_title(title)
        fig.savefig(os.path.join(directory, fname))
        ax.cla()


### Not modified for new data type

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

def frames_to_mp4(directory, prefix='Loop', outname='out'):
    # Send command to create video with ffmpeg
    # TODO: have it recognize the file prefix
    cmd = (r'cd "{}" & ffmpeg -framerate 10 -i {}_%03d.png -c:v libx264 '
            '-r 15 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
            '{}.mp4').format(directory, prefix, outname)
    os.system(cmd)
