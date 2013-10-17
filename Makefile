

PYTHONDIR=`python-config --includes`
NPDIR=`python-config --prefix`/Extras/lib/python/numpy/core/include/

all: lib/cell_model_opt.so

lib/cell_model_opt.so: lib/cell_model_opt.c
	gcc -shared -lpython2.6 -o lib/cell_model_opt.so -fPIC -O2 -I$(PYTHONDIR) -I$(NPDIR) lib/cell_model_opt.c

lib/cell_model_opt.c: src/cell_model_opt.pyx
	cython -o lib/cell_model_opt.c src/cell_model_opt.pyx
