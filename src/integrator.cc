#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "integrator.cuh"

namespace py = pybind11;

auto stencil_as_dict(const Stencil& stencil)
{
    py::dict dict;
    dict["dt"] = stencil.dt;
    dict["dx"] = stencil.dx;
    dict["dy"] = stencil.dy;
    return dict;
};

auto model_as_dict(const Model& model)
{
    py::dict dict;
    dict["a"] = model.a;
    dict["b"] = model.b;
    dict["c"] = model.c;
    dict["kappa"] = model.kappa;
    dict["lamb"] = model.lambda;
    dict["zeta"] = model.zeta;
    return dict;
};

PYBIND11_MODULE(integrator, m)
{
    py::register_exception<kernel::CudaError>(m, "CudaException");

    {
        py::class_<Model> py_class(m, "Model");
        py_class.def(py::init<>())
                .def(py::init<Scalar, Scalar, Scalar, Scalar, Scalar, Scalar>())
                .def(py::init<Model>())
                .def_readwrite("a", &Model::a)
                .def_readwrite("b", &Model::b)
                .def_readwrite("c", &Model::c)
                .def_readwrite("kappa", &Model::kappa)
                .def_readwrite("lamb", &Model::lambda)
                .def_readwrite("zeta", &Model::zeta)
                .def("as_dict", &model_as_dict)
                .def("as_tuple",
                    [](const Model& model)
                    {
                        return model.as_tuple();
                    })
                .def("__str__",
                    [](const Model& model)
                    {
                        return py::str(model_as_dict(model));
                    })
                .def("__repr__",
                    [](const Model& model)
                    {
                        return py::str("<Model ") + py::str(model_as_dict(model)) + py::str(">");
                    });
    }

    {
        py::class_<Stencil> py_class(m, "Stencil");
        py_class.def(py::init<>())
                .def(py::init<Scalar, Scalar, Scalar>())
                .def(py::init<Stencil>())
                .def_readwrite("dt", &Stencil::dt)
                .def_readwrite("dx", &Stencil::dx)
                .def_readwrite("dy", &Stencil::dy)
                .def("as_dict", &stencil_as_dict)
                .def("as_tuple",
                    [](const Stencil& stencil)
                    {
                        return stencil.as_tuple();
                    })
                .def("__str__",
                    [](const Stencil& stencil)
                    {
                        return py::str(stencil_as_dict(stencil));
                    })
                .def("__repr__",
                    [](const Stencil& stencil)
                    {
                        return py::str("<Stencil ") + py::str(stencil_as_dict(stencil)) + py::str(">");
                    });
    }
}
