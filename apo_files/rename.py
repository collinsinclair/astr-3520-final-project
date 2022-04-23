import os


# renames the files in this directory
# from the pattern xxxx.####.fits to ####.xxxx.fits
# example: BDFlux.0040.fits -> 0040.BDFlux.fits

def rename(files):
    for f in files:
        if f.endswith('.fits'):
            fname = f.split('.')
            if len(fname) == 3 and fname[2] == 'fits':
                newname = fname[1] + '.' + fname[0] + '.fits'
                os.rename(f, newname)


if __name__ == '__main__':
    files = os.listdir()
    rename(files)
