import astropy.io.fits as pyf
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import convolve, Box1DKernel
from scipy.interpolate import UnivariateSpline
from astropy.stats import SigmaClip

datafolder = "/Users/collinsinclair/APO/Q1CU01/UT220311/"

# much of this is adapted from original work by James Davenport at UW, in pyDIS, kosmos

kosmosINFO = {'BIN11': "[1:1024,1:4096];[1025:2048,1:4096]",
              'BIN12': "[1:1024,1:2048];[1025:2048,1:2048]",
              'BIN21': "[1:512,1:4096];[513:1024,1:4096]",
              'BIN22': "[1:512,1:2048];[513:1024,1:2048]"}


def ccd_trim(hdu_i, silent=True):
    """Apply Trimming of overscan regions of DIS or KOSMOS"""

    if hdu_i[0].header['INSTRUME'] == 'DIS':
        if silent is False:
            print("Trimming for instrument " + hdu_i[0].header['INSTRUME'])
        datasec = hdu_i[0].header['DATASEC'][1:-1].replace(':', ',').split(',')
        d = list(map(int, datasec))
        im_i = hdu_i[0].data[d[2] - 1:d[3], d[0] - 1:d[1]]
    elif hdu_i[0].header['INSTRUME'] == 'kosmos':
        if silent is False:
            print("Trimming for instrument " + hdu_i[0].header['INSTRUME'])
        sel = "BIN{0}{1}".format(hdu_i[0].header['BINX'], hdu_i[0].header['BINY'])
        datasec = kosmosINFO[sel]
        ds_l, ds_r = datasec.replace(':', ',').split(';')
        i_d_l = list(map(int, ds_l[1:-1].split(',')))
        i_d_r = list(map(int, ds_r[1:-1].split(',')))

        im_i_l = hdu_i[0].data[i_d_l[2] - 1:i_d_l[3], i_d_l[0] - 1:i_d_l[1]]
        im_i_r = hdu_i[0].data[i_d_r[2] - 1:i_d_r[3], i_d_r[0] - 1:i_d_r[1]]

        # nxL , nyL = im_i_l.shape
        # nxR , nyR = im_i_r.shape

        im_i = np.append(im_i_l, im_i_r, axis=1)
    else:
        print("ERROR: mismatched instruments? Check instrument header, should be in ['kosmos','DIS']")
        return

    return im_i


def bias_combine(biaslist, output="../ccd_calibrations/BIAS.fits", Trim=True, Silent=True):
    """
    Combine the bias frames in to a master bias image. Only applies median combine

    Parameters
    ----------
    biaslist : list
        input list of strings with paths to individual raw bias fits images, e.g., ['bias.001.fits','bias.002.fits','bias.003.fits']
    output: str, optional
        Name of the master bias image to write. (Default is "BIAS.fits")
    Trim : bool, optional
        Trim the image using the DATASEC keyword or info for KOSMOS
    Silent : bool, optional
        If False, print details about the biascombine. (Default is True)

    Returns
    -------
    bias : 2-d array
        The median combined master bias image
    """

    nfiles = len(biaslist)

    if Silent is False:
        print("BiasCombine: Combining {0} files --- \n".format(nfiles))
        for i in range(nfiles):
            print(biaslist[i])

        print("Saving and overwriting fits file as " + output)

    for i in range(0, nfiles):
        hdu_i = pyf.open(biaslist[i])
        hdu_i.verify('silentfix')

        if Trim is False:
            im_i = hdu_i[0].data
        else:  # if Trim is True
            im_i = ccd_trim(hdu_i, silent=Silent)

        # create image stack
        if i == 0:
            all_data = im_i
            header0 = hdu_i[0].header
        else:  # elif i > 0:
            all_data = np.dstack((all_data, im_i))
        hdu_i.close(closed=True)

    # do median across whole stack
    bias = np.nanmedian(all_data, axis=2)

    # make comment in header

    strlist = ""
    for i in range(nfiles):
        strlist = strlist + ", " + biaslist[i]
    comment = "Header info is based on first file used for median combine: " + strlist
    header0["COMMENT"] = comment

    # write output to disk for later use
    hduOut = pyf.PrimaryHDU(bias, header=header0)
    hduOut.writeto(output, overwrite=True)
    return bias


def ArcCombine(arclist, inputbias=None, output="../ccd_calibrations/ARC.fits", Trim=True, Silent=True):
    """
    Combine the arc frames, adding them together. If only single arc is listed, does the trimming.

    Parameters
    ----------
    arclist : list
        input list of strings with paths to individual raw bias fits images, e.g., ['arc.001.fits','arc.002.fits','arc.003.fits']
    inputbias: str or numpy array
        if a string, the file is opened as a fits file, are simply the array is taken as is
    output: str, optional
        Name of the summed arc image to write. (Default is "ARC.fits")
    trim : bool, optional
        Trim the image using the DATASEC keyword or info for KOSMOS
    silent : bool, optional
        If False, print details about the biascombine. (Default is True)

    Returns
    -------
    arcout : 2-d array
        The summed arc image
    """

    if inputbias is not None:
        if Silent is False:
            print("Using bias subtraction...")
        if isinstance(inputbias, str):
            hdu_i = pyf.open(inputbias)
            biasM = hdu_i[0].data
        elif isinstance(inputbias, np.ndarray):
            biasM = inputbias
        else:
            print("Warning: check bias inputs, returning None")
            return None
    else:
        biasM = 0.

    if isinstance(arclist, str):
        hdu_i = pyf.open(arclist)
        hdu_i.verify('silentfix')

        if Trim is False:
            im_i = hdu_i[0].data
        if Trim is True:
            im_i = ccd_trim(hdu_i, silent=Silent)

        arcout = im_i - biasM
        header0 = hdu_i[0].header
        comment = "Trimmed Arc file with bias subtraction: " + arclist

    elif isinstance(arclist, list):
        nfiles = len(arclist)
        if nfiles == 1:
            hdu_i = pyf.open(arclist[0])
            hdu_i.verify('silentfix')

            if Trim is False:
                im_i = hdu_i[0].data
            if Trim is True:
                im_i = ccd_trim(hdu_i, silent=Silent)

            arcout = im_i - biasM
            header0 = hdu_i[0].header
            comment = "Trimmed Arc file with bias subtraction: " + arclist[0]

        elif nfiles > 1:

            if Silent is False:
                print("ArcCombine with bias subtraction: Summing {0} files --- \n".format(nfiles))
                for i in range(nfiles):
                    print(arclist[i])

                print("Saving and overwriting fits file as " + output)

            for i in range(0, nfiles):
                hdu_i = pyf.open(arclist[i])
                hdu_i.verify('silentfix')

                if Trim is False:
                    im_i = hdu_i[0].data
                if Trim is True:
                    im_i = ccd_trim(hdu_i, silent=Silent)

                # create image stack
                if (i == 0):
                    all_data = im_i - biasM
                    header0 = hdu_i[0].header
                elif (i > 0):
                    all_data = np.dstack((all_data, im_i - biasM))
                    hdu_i.close(closed=True)

            arcout = all_data.sum(axis=2)
            strlist = ""
            for i in range(nfiles):
                strlist = strlist + ", " + arclist[i]
            comment = "Header info is based on first file used for summation: " + strlist

    else:
        print("ERROR: Expected input to be a list of strings, i.e., ['file1','file2'] ")
        return None

    ## make comment in header

    header0["COMMENT"] = comment

    # write output to disk for later use
    hduOut = pyf.PrimaryHDU(arcout, header=header0)
    hduOut.writeto(output, overwrite=True)
    return arcout


def FlatCombine(flatlist, inputbias, output='../ccd_calibrations/FLAT.fits', Trim=True, mode='spline', flat_poly=7,
                Display=True, response=True, SpAxis=0, badmask=None, Silent=True):
    """
    Combine the flat frames in to a master flat image. Subtracts the
    master bias image first from each flat image. Currently only
    supports median combining the images.

    Parameters
    ----------
    flatlist : list[str]
        list of strings, filenames for each flat to combine
    inputbias : str or 2-d array
        Either the path to the master bias image (str) or
        the output from 2-d array output from BiasCombine
    output : str, optional
        Name of the master flat image to write. (Default is "FLAT.fits")
    response : bool, optional
        If set to True, first combines the median image stack along the
        spatial direction, then fits spine to 1D curve, then
        divides each row in flat by this structure. This nominally divides
        out the spectrum of the flat field lamp. (Default is True)
    mode: str, optional
        Default is 'spline'; but can also be 'poly' -- defines the method of fitting used to determine the 1d flat curve when response = True. If 'spline', the 'flat_poly' keyword is not used. 'spline' uses the UnivariateSpline method of scipy.interpolate with ext=0, k=2 ,s=0.001
    trim : bool, optional
        Trim the image using the DATASEC keyword in the header, employs CCDTrim
    Display : bool, optional
        Set to True to show 1d flat, and final flat (Default is False)
    flat_poly : int, optional
        Polynomial order to fit 1d flat curve with. Only used if
        response is set to True. (Default is 5)
    SpAxis : int, optional
        Set which axis the spatial dimension is along. 1 = columns, 0 = rows.
        (Default is 0)
    badmask: str or 2d-array, optional
        Used to mask out bad pixels in generation of flat, requires pre-existing badpixel mask

    Returns
    -------
    flat : 2-d array
        The median combined master flat
    """
    # read the bias in, BUT we don't know if it's the numpy array or file name
    if isinstance(inputbias, str):
        # read in file if a string
        bias_im = pyf.open(inputbias)[0].data
    elif isinstance(inputbias, np.ndarray):
        # assume is proper array from BiasCombine function
        bias_im = inputbias
    else:
        print(
            "ERROR: inputbias not of correct type, should be string to fits filename, or 2D numpy array of the maste bias image.")
        return None

    # check for a badpixel mask,  True (1) for bad pixels False (0) for good pixels
    if badmask is not None:
        if isinstance(badmask, str):
            # read in file if a string
            badpix = pyf.open(badmask)[0].data
        elif isinstance(badmask, np.ndarray):
            # assume it is proper array
            badpix = badmask

    # flatlist is list of filenames

    if SpAxis == 1:
        DispAxis = 0
    elif SpAxis == 0:
        DispAxis = 1
    else:
        print("ERROR: SpAxis needs to be 1 for columns or 0 for rows (integers)")
        return None

    nfiles = len(flatlist)  # input should be list of flat files

    for i in range(0, nfiles):
        hdu_i = pyf.open(flatlist[i])
        hdu_i.verify('silentfix')
        if Trim is False:
            im_i = hdu_i[0].data - bias_im
        elif Trim is True:
            im_i = ccd_trim(hdu_i, silent=Silent) - bias_im

        if badmask is not None:
            im_i[badpix.astype(
                'bool')] = np.nan  # use NaN to mask out bad pixels, need to make sure input mask is the right shape, trimmed or not

        # check for bad regions (not illuminated) in the spatial direction
        ycomp = np.nansum(im_i, axis=DispAxis)  # compress to spatial axis only
        illum_thresh = 0.8  # value compressed data must reach to be used for flat normalization
        ok = np.where((ycomp >= np.nanmedian(ycomp) * illum_thresh))

        # assume a median scaling for each flat to account for possible different exposure times
        if (i == 0):
            header0 = hdu_i[0].header
            if SpAxis == 0:
                all_data = im_i / np.nanmedian(im_i[ok, :])
            elif SpAxis == 1:
                all_data = im_i / np.nanmedian(im_i[:, ok])

        elif (i > 0):

            if SpAxis == 0:
                all_data = np.dstack((all_data, im_i / np.nanmedian(im_i[ok, :])))
            elif SpAxis == 1:
                all_data = np.dstack((all_data, im_i / np.nanmedian(im_i[:, ok])))

        hdu_i.close(closed=True)

    # do median across whole stack of flat images
    flat_stack = np.nanmedian(all_data, axis=2)
    # pdb.set_trace()
    # define the wavelength axis

    if response is True:
        # fit along dispersion direction
        fltshape = flat_stack.shape

        xdata = np.arange(fltshape[DispAxis])  # x pixels, matching dispersion direction

        # median along spatial axis, smooth w/ 5pixel boxcar, take log of flux; NaNs ignored in convolve
        flat_1d = np.log10(convolve(np.nanmedian(flat_stack, axis=SpAxis), Box1DKernel(5)))

        if mode == 'spline':
            spl = UnivariateSpline(xdata, flat_1d, ext=0, k=2, s=0.001)
            flat_curve = 10.0 ** spl(xdata)
        elif mode == 'poly':
            # fit log flux with polynomial
            flat_fit = np.polyfit(xdata, flat_1d, flat_poly)
            # get rid of log
            flat_curve = 10.0 ** np.polyval(flat_fit, xdata)

        if Display is True:
            plt.figure()
            plt.plot(10.0 ** flat_1d)
            plt.plot(xdata, flat_curve, 'r')
            plt.show()

        # divide median stacked flat by this RESPONSE curve
        flat = np.zeros_like(flat_stack)

        if SpAxis == 0:
            for i in range(fltshape[SpAxis]):
                flat[i, :] = flat_stack[i, :] / flat_curve
        elif SpAxis == 1:
            for i in range(fltshape[SpAxis]):
                flat[:, i] = flat_stack[:, i] / flat_curve
    else:
        flat = flat_stack

    if Display is True:
        plt.figure()
        plt.imshow(flat, origin='lower', aspect='auto')
        plt.colorbar()
        plt.show()

    strlist = ""
    for i in range(nfiles):
        strlist = strlist + ", " + flatlist[i]
    comment = "Header info is based on first file used for median combine of flats: " + strlist
    header0["COMMENT"] = comment

    # write output to disk for later use; add fits entries for masks to allow loading flat and mask w/o running flatcombine again
    hduOut = pyf.PrimaryHDU(flat, header=header0)
    ilumfmask = pyf.ImageHDU(ok[0], name="FlatMask")

    # place holder for actual bad pixel mask -- 1/True in mask is invalid data/pixel -- use to replace bad data with NaNs, need flat to already exist to match shape
    if badmask is None:
        badpix = np.zeros_like(flat)

    badout = pyf.ImageHDU(badpix, name="BadMask")

    hduL = pyf.HDUList([hduOut, ilumfmask, badout])  # use HDUList as container for fits image + masks
    hduL.writeto(output, overwrite=True)

    return flat, ok[0], badpix


image = pyf.open("../apo_files/0002.bias.fits")
bias_l = image[0].data[0:4072, 0:1023]
bias_r = image[0].data[0:4072, 1024:2048]
overscan_l = image[0].data[0:4072, 2048:2100]
overscan_r = image[0].data[0:4072, 2100:2146]

filterout = SigmaClip(sigma=5)  # set up outlier rejection for hot pixels etc.

dat_1 = bias_l

datF_1 = filterout(dat_1)  # apply outlier filter
binsE_1 = np.arange(datF_1.min(), datF_1.max() + 1) - 0.5  # sets up histogram binning

outH_1 = np.histogram(datF_1.data[~datF_1.mask],
                      bins=binsE_1)  # uses numpy to create histrogram of all pixel counts in bias 2D array
xh_1 = outH_1[1][0:-1] + 0.5
yh_1 = outH_1[0]

dat_2 = bias_r

datF_2 = filterout(dat_2)  # apply outlier filter
binsE_2 = np.arange(datF_2.min(), datF_2.max() + 1) - 0.5  # sets up histogram binning

outH_2 = np.histogram(datF_2.data[~datF_2.mask],
                      bins=binsE_2)  # uses numpy to create histrogram of all pixel counts in bias 2D array
xh_2 = outH_2[1][0:-1] + 0.5
yh_2 = outH_2[0]

dat_3 = overscan_l

datF_3 = filterout(dat_3)  # apply outlier filter
binsE_3 = np.arange(datF_3.min(), datF_3.max() + 1) - 0.5  ## sets up histogram binning

outH_3 = np.histogram(datF_3.data[~datF_3.mask],
                      bins=binsE_3)  ## uses numpy to create histrogram of all pixel counts in bias 2D array
xh_3 = outH_3[1][0:-1] + 0.5
yh_3 = outH_3[0]

dat_4 = overscan_r

datF_4 = filterout(dat_4)  # apply outlier filter
binsE_4 = np.arange(datF_4.min(), datF_4.max() + 1) - 0.5  # sets up histogram binning

outH_4 = np.histogram(datF_4.data[~datF_4.mask],
                      bins=binsE_4)  # uses numpy to create histrogram of all pixel counts in bias 2D array
xh_4 = outH_4[1][0:-1] + 0.5
yh_4 = outH_4[0]

# Plotting  =------------------

figH, axH = plt.subplots()
axH.step(xh_1, yh_1, where='mid', label='Left Data')
axH.step(xh_2, yh_2, where='mid', label='Right Data')
axH.step(xh_3, yh_3, where='mid', label='Left Overscan')
axH.step(xh_4, yh_4, where='mid', label='Right Overscan')

axH.set_xlabel("Bias Level")
axH.set_ylabel("Histogram Counts")
axH.legend()
plt.savefig("../plots/data-vs-overscan.png", dpi=300)

blist = ["../apo_files/0002.bias.fits", "../apo_files/0003.bias.fits", "../apo_files/0004.bias.fits",
         "../apo_files/0005.bias.fits", "../apo_files/0006.bias.fits"]

biasname = "../ccd_calibrations/MasterBias.fits"

biasM = bias_combine(blist, output=biasname, Trim=True, Silent=False)

flatlist = ['../apo_files/0043.Flat.fits', '../apo_files/0044.Flat.fits', '../apo_files/0045.Flat.fits']

flatname = "../ccd_calibrations/MasterFlat.fits"  ## do this independently for DIS Red and DIS Blue (change names) -- you need separate master biases for each CCD, with different saved files

# be careful about these filenames as the default mode is to overwrite a file with the same name
# inputbias should point to the bias file we used before (make sure you use the proper ones for DIS red/blue)


## --- Fill in SpAxis with 1 for columns as the spatial axis, and 0 for rows as the spatial axis --- KOSMOS and DIS are different


FlatM, illumindex, badpix = FlatCombine(flatlist, biasname, output=flatname, SpAxis=1,
                                        Trim=True, Silent=True, Display=False, mode='spline')

# process images with master bias and flat

comp_lamp_files = ['../apo_files/0033.lamps.fits', '../apo_files/0046.Lamps.fits']
comp_lamp_out_names = ['../ccd_calibrations/comp_1.fits', '../ccd_calibrations/comp_2.fits']
flat_file = "../ccd_calibrations/MasterFlat.fits"
bias_file = "../ccd_calibrations/MasterBias.fits"

for inFile, outFile in zip(comp_lamp_files, comp_lamp_out_names):
    # out = ((in - masterbias) / masterflat) * gain

    # read in the lamp image
    lamp = pyf.open(inFile)
    lamp_data = lamp[0].data[:, 0:2048]

    # read in the master flat
    flat = pyf.open(flat_file)
    flat_data = flat[0].data

    # read in the master bias
    bias = pyf.open(bias_file)
    bias_data = bias[0].data

    # subtract the master bias
    lamp_data = lamp_data - bias_data

    # divide by the master flat
    lamp_data = lamp_data / flat_data

    # multiply by the gain
    gain = 0.6  # e-/ADU
    lamp_data = lamp_data * gain

    # write the output file
    pyf.writeto(outFile, lamp_data, lamp[0].header, overwrite=True)

bd_flux = ['../apo_files/0040.BDFlux.fits', '../apo_files/0041.BDFlux.fits', '../apo_files/0042.BDFlux.fits']
bd_flux_out = ['../ccd_calibrations/bd_flux_1.fits', '../ccd_calibrations/bd_flux_2.fits',
               '../ccd_calibrations/bd_flux_3.fits']

for inFile, outFile in zip(bd_flux, bd_flux_out):
    # out = ((in - masterbias) / masterflat) * gain

    # read in the lamp image
    lamp = pyf.open(inFile)
    lamp_data = lamp[0].data[:, 300:1600]
    print("min bd_flux = ", np.min(lamp_data))

    # read in the master flat
    flat = pyf.open(flat_file)
    flat_data = flat[0].data[:, 300:1600]
    print("min flat = ", np.min(flat_data))

    # read in the master bias
    bias = pyf.open(bias_file)
    # bias_data = bias[0].data[:, 300:1600]
    bias_data = np.min(lamp_data) * np.ones(lamp_data.shape)
    print("min bias = ", np.min(bias_data))

    # subtract the master bias
    lamp_data = lamp_data - bias_data
    print("min bd_flux - bias = ", np.min(lamp_data))

    # divide by the master flat
    lamp_data = lamp_data / flat_data
    print("min bd_flux / flat = ", np.min(lamp_data))

    # multiply by the gain
    gain = 0.6  # e-/ADU
    lamp_data = lamp_data * gain
    print("min bd_flux * gain = ", np.min(lamp_data))

    # write the output file
    pyf.writeto(outFile, lamp_data, lamp[0].header, overwrite=True)

science_images = ['../apo_files/0032.SUAur.fits', '../apo_files/0036.DRTau.fits', '../apo_files/0039.RWAur.fits']
science_out = ['../ccd_calibrations/suaur.fits', '../ccd_calibrations/drtau.fits', '../ccd_calibrations/rwaur.fits']
for inFile, outFile in zip(science_images, science_out):
    # out = ((in - masterbias) / masterflat) * gain

    # read in the lamp image
    lamp = pyf.open(inFile)
    lamp_data = lamp[0].data[:, 300:1600]
    print("min bd_flux = ", np.min(lamp_data))

    # read in the master flat
    flat = pyf.open(flat_file)
    flat_data = flat[0].data[:, 300:1600]
    print("min flat = ", np.min(flat_data))

    # read in the master bias
    bias = pyf.open(bias_file)
    # bias_data = bias[0].data[:, 300:1600]
    bias_data = np.min(lamp_data) * np.ones(lamp_data.shape)
    print("min bias = ", np.min(bias_data))

    # subtract the master bias
    lamp_data = lamp_data - bias_data
    print("min bd_flux - bias = ", np.min(lamp_data))

    # divide by the master flat
    lamp_data = lamp_data / flat_data
    print("min bd_flux / flat = ", np.min(lamp_data))

    # multiply by the gain
    gain = 0.6  # e-/ADU
    lamp_data = lamp_data * gain
    print("min bd_flux * gain = ", np.min(lamp_data))

    # write the output file
    pyf.writeto(outFile, lamp_data, lamp[0].header, overwrite=True)
