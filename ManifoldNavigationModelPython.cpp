/*
<%
setup_pybind11(cfg)
cfg['compiler_args'] = ['-std=c++11', '-fopenmp', '-I./libigl/include']
cfg['linker_args'] = ['-fopenmp']
cfg['sources'] = ['model.cpp', 'Mesh.cpp']
%>
*/

#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include "model.h"
namespace py = pybind11;


PYBIND11_MODULE(ManifoldNavigationModelPython, m) {    
    py::class_<Model>(m, "ManifoldNavigationModelPython")
        .def(py::init<std::string, int>())
        .def("getCurrentState", &Model::getPosition)
        .def("setInitialState", &Model::setPosition)
        .def("findClosestFace", &Model::findClosestFace)
        .def("reset", &Model::createInitialState)
        .def("step", &Model::step_given_field);

}