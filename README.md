# Estimation of the heavy-element content (metallicity) of fundamental-mode RR Lyrae stars from their  light curves in the Gaia _G_ and VISTA _Ks_ wavebands

This is a Python3 library for the metallicity estimation of RR Lyrae stars 
by deep learning, using the method published by Dekany & Grebel (2022).

## Installation

Simply use `git clone <URL>` to clone this library into a local directory. 
Subsequently, it can be added to your `PYTHONPATH` environment variable
either temporarily by issuing the `sys.path.append("/your/local/directory")` 
command in python, or permanantly by exporting your directory into 
`PYTHONPATH`.
For example, if using the bash shell, add the 
`export PYTHONPATH="${PYTHONPATH}:/your/local/directory"` to the content of the 
~/.bashrc file.

Optionally, extract the gzipped tar archives containing the development and target 
data sets by issuing the following commands:

`cat lc_target_g.tar.gz.* | tar xzvf -`

`tar xzvf lc_dev_?.tar.gz`

`tar xvzf lc_target_k.tar.gz`

### Dependencies

This library was developed and tested in the following python environment:

- python 3.8.10
- tensorflow 2.5.0
- numpy 1.19.5
- pandas 1.3.0
- joblib 1.0.1
- scipy 1.6.3

We suggest to use this library in the same environment, created by a virtual environment manager 
such as [conda](https://docs.conda.io/en/latest/).

## Usage
The program for training and/or deploying the neural network can be run by:

`python rrlfeh_nn.py [OPTIONS]`

or by supplying a parameter file that contains the command-line arguments:

`python rrlfeh_nn.py @<parameter_file>`

The full list of command-line options can be printed on the STDOUT by:

`python rrlfeh_nn.py --help`

Example parameter files `gfeh.par` and `kfeh.par` are provided for the 
_G_- and _Ks_-band data and models, respectively. The data and parameter settings
in the above files are equivalent to those used in the corresponding 
[paper]().

The default settings assume a (multi-) GPU hardware environment. Use the -cpu
option to run the program on the CPU, although this is not recommended (slow)
for training.

### Input

The program reads in a list of photometric time series (light curves) 
stored in separate files located in subdirectories, specified by the 
`--input_file <filename>` and `--lcdir <dirname>` parameters. 
The list file must contain a header preceded by a `#` character,
and have at least two fields: `id` and `period`, the latter being the
corresponding star's pulsation period, and `id` its identifier.
The light curve files for each star id will be looked for as
`<rootdir>/<lcdir>/<id><suffix>`. The suffix can be specified by the 
`--lcfile_suffices` parameter, the arguments of which must match those 
of the `--wavebands` parameter (in the corresponding paper, 
single wavebands are used).

The column names to be used can be passed to the program by the 
`--columns <colnames>` parameter, and a subset of the data can be selected by 
specifying a set of criteria on these (meta)data by the `--subset <expression>`
parameter, whose argument must follow the pandas.dataframe.query format
(see the example parameter files for examples).

With the `--meta_input <colnames>` option, one can specify metadata 
stored in the input list file's columns to be fed into
the neural network in a second input layer after its recurrent/convolutional
encoder part. Note that an appropriate model must be selected for this,
and the original analysis did not use such an architecture.

The light curve files must contain
two columns: time and magnitude (brightness). The periodic light curves 
must be phase-folded and phase-aligned by the first Fourier phase made equal to zero.
The [lcfit](https://github.com/idekany/lcfit) program can be used 
for preparing further time series in this way for the input of the 
neural network.

### Output

When deployed on a target data set using the `--predict`
option, the program writes out the predicted [Fe/H] 
metallicity indices to a file specified by the 
`--target_output_file <filename>` option.

When the neural network is trained, a set of files are generated in a subdirectory
specified by the `--outdir` parameter, including the checkpoint and/or final weights,
the serialized model, performance plots, histograms, etc.
A live performance plot `liveplot.pdf` is generated in the `<rootdir>` for monitoring
purposes, its properties can be modified with the `--n_zoom` and `--n_update` options
(see the command-line help for more details).

If the `--refit` option is set, the model is refitted on the entire development
data set with the best-performing hyperparameter setting found by cross-validation,
yielding a 'monolithic' single final model. Otherwise, 
a k-fold model ensemble is generated.

Please invoke the built-in documentation by `python rrlfeh_nn.py --help` for more
details/options.

## License

[MIT](https://choosealicense.com/licenses/mit/), see `LICENSE`.
