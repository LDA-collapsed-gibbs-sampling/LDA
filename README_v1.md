Install the dependencies using anaconda, use the anaconda.

conda env create --file requirements_conda.txt

To build lda from the source, use
make cython
python setup.py build_ext --inplace

To run the main.py file, use 
python main.py
