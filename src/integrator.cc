#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "integrator.h"

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
    dict["temperature"] = model.temperature;
    return dict;
};

PYBIND11_MODULE(integrator, m)
{
    py::register_exception<kernel::CudaError>(m, "CudaException");

    {
        py::class_<Model> py_class(m, "Model");
        py_class.def(py::init<>())
                .def(py::init<Scalar, Scalar, Scalar, Scalar,
                              Scalar, Scalar, Scalar>(),
                     py::arg("a"), py::arg("b"), py::arg("c"),
                     py::arg("kappa"), py::arg("lamb"), py::arg("zeta"),
                     py::arg("T"))
                .def(py::init<Model>(), py::arg("model"))
                .def_readwrite("a", &Model::a)
                .def_readwrite("b", &Model::b)
                .def_readwrite("c", &Model::c)
                .def_readwrite("kappa", &Model::kappa)
                .def_readwrite("lamb", &Model::lambda)
                .def_readwrite("zeta", &Model::zeta)
                .def_readwrite("T", &Model::temperature)
                .def("as_dict", &model_as_dict)
                .def("as_tuple",
                    [](const Model& model)
                    {
                        return model.as_tuple();
                    })
                .def("__eq__",
                    [](const Model& a, const Model& b)
                    {
                        return a == b;
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
                .def(py::init<Scalar, Scalar, Scalar>(),
                     py::arg("dt"), py::arg("dx"), py::arg("dy"))
                .def(py::init<Stencil>(), py::arg("stencil"))
                .def_readwrite("dt", &Stencil::dt)
                .def_readwrite("dx", &Stencil::dx)
                .def_readwrite("dy", &Stencil::dy)
                .def("as_dict", &stencil_as_dict)
                .def("as_tuple",
                    [](const Stencil& stencil)
                    {
                        return stencil.as_tuple();
                    })
                .def("__eq__",
                    [](const Stencil& a, const Stencil& b)
                    {
                        return a == b;
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

    {
        py::class_<Integrator> py_class(m, "Integrator");
        py_class.def(py::init<const HostFieldRef&, Stencil, Model>(),
                     py::arg("field"), py::arg("stencil"), py::arg("model"))
                .def_property_readonly("stencil", &Integrator::get_stencil, py::return_value_policy::move)
                .def_property_readonly("model", &Integrator::get_model, py::return_value_policy::move)
                .def_property_readonly("field", &Integrator::get_field, py::return_value_policy::move)
                .def_property_readonly("current", &Integrator::get_current, py::return_value_policy::move)
                .def_property_readonly("passive_current", &Integrator::get_passive_current, py::return_value_policy::move)
                .def_property_readonly("active_current", &Integrator::get_active_current, py::return_value_policy::move)
                .def_property_readonly("lambda_current", &Integrator::get_lambda_current, py::return_value_policy::move)
                .def_property_readonly("circulating_current", &Integrator::get_circulating_current, py::return_value_policy::move)
                .def_property_readonly("random_current", &Integrator::get_random_current, py::return_value_policy::move)
                .def_property_readonly("time", &Integrator::get_time)
                .def_property_readonly("timestep", &Integrator::get_timestep)
                .def("run", &Integrator::run, py::arg("nsteps"));
    }
}
