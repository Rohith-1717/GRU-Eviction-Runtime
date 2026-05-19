#include <pybind11/pybind11.h>
#include "FaultHandler.hpp"
#include "MemRegion.hpp"
#include "MemoryManager.hpp"

namespace py = pybind11;

PYBIND11_MODULE(swapcore_native, m){
    py::class_<MemoryManager>(m, "MemoryManager")
        .def(py::init<>())
        .def("faultLatencyNs", &MemoryManager::faultLatencyNs)
        .def("swapWriteLatencyNs", &MemoryManager::swapWriteLatencyNs)
        .def("swapReadLatencyNs", &MemoryManager::swapReadLatencyNs);

    py::class_<FaultHandler>(m, "FaultHandler")
        .def(py::init<MemoryManager*>())
        .def("start", &FaultHandler::start)
        .def("stop", &FaultHandler::stop);

    py::class_<MemRegion>(m, "MemRegion")
        .def(py::init<size_t, FaultHandler*>())
        .def("get", &MemRegion::get)
        .def("set", &MemRegion::set)
        .def("getSize", &MemRegion::getSize);
}