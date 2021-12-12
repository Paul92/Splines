# Differential Splines library

A different implementation of the BBS toolbox of Florent Brunet
(brnt.eu/downloads.php).

The goal is to provide a minimal interface for 2D image warps from one
reference frame and multiple target frames, which can be used both in C++ and
Python.

The main class is Warp, which uses BBS internally. See `src/warps.h` for
documentation and the `examples` directory.

## Installation

The C++ implementation is based on SuiteSparse. `cmake` is provided as build
system. `pybind11` is used for the Python wrapper.



