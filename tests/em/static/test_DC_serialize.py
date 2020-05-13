from __future__ import print_function
import unittest
import numpy as np
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG import maps
from discretize import TensorMesh
import properties

class SerializationTest(unittest.TestCase):

    receivers_to_test = [
        "Pole",
        "Dipole",
    ]

    sources_to_test = [
        'Pole',
        'Dipole',
    ]

    simulations_to_test = [
        "Simulation3DCellCentered",
        "Simulation3DNodal",
        "Simulation2DCellCentered",
        "Simulation2DNodal",
        "Simulation1DLayers"
    ]

    def test_receiver_serialization(self):
        locationsM = np.atleast_2d([1, 2, 3])
        locationsN = np.atleast_2d([2, 3, 4])

        for r in self.receivers_to_test:
            print(r)
            if r == 'Dipole':
                rx = dc.receivers.Dipole(locationsM, locationsN)
            else:
                rx = dc.receivers.Pole(locationsM)
            rxs = rx.serialize()
            rx2 = dc.receivers.BaseRx.deserialize(rxs, trusted=True)

            # assert that they are not the same object in memory
            self.assertTrue(rx is not rx2)
            self.assertTrue(properties.equal(rx, rx2))

    def test_source_serialization(self):
        locationA = np.atleast_2d([0, 0, 0])
        locationB = np.atleast_2d([1, 1, 1])

        locationsM = np.atleast_2d([1, 2, 3])
        locationsN = np.atleast_2d([2, 3, 4])

        rxP = dc.receivers.Pole(locationsM)
        rxD = dc.receivers.Dipole(locationsM, locationsN)

        for s in self.sources_to_test:
            if s == 'Dipole':
                src = dc.sources.Dipole([rxP, rxD], locationA=locationA, locationB=locationB)
            else:
                src = dc.sources.Dipole([rxP, rxD], locationA=locationA, locationB=locationB)
            src_s = src.serialize()

            src2 = dc.sources.BaseFDEMSrc.deserialize(
                src_s, trusted=True
            )

            # assert that they are not the same object in memory
            self.assertTrue(src is not src2)
            self.assertTrue(src.receiver_list[0] is not src2.receiver_list[0])
            self.assertTrue(src.receiver_list[1] is not src2.receiver_list[1])
            self.assertTrue(properties.equal(src, src2))

    def test_survey_serialization(self):
        locationA = np.array([0, 0, 0])
        locationB = np.array([1, 1, 1])

        locationsM = np.atleast_2d([1, 2, 3])
        locationsN = np.atleast_2d([2, 3, 4])

        rxP = dc.receivers.Pole(locationsM)
        rxD = dc.receivers.Dipole(locationsM, locationsN)

        srcP = dc.sources.Pole([rxP, rxD], locationA)
        srcD = dc.sources.Dipole([rxP, rxD], locationA, locationB)

        survey = dc.Survey([srcP, srcD])

        survey2 = dc.Survey.deserialize(survey.serialize(), trusted=True)

        self.assertTrue(survey is not survey2)
        self.assertTrue(properties.equal(survey, survey2))


    def test_simulation_serialization(self):
        mesh = TensorMesh([np.ones(10), np.ones(10), np.ones(10)])
        sigma_map = maps.ExpMap(mesh)
        sigma = np.ones(mesh.nC)

        thicks = np.random.rand(mesh.nC-1)

        locationA = np.array([0, 0, 0])
        locationB = np.array([1, 1, 1])

        locationsM = np.atleast_2d([1, 2, 3])
        locationsN = np.atleast_2d([2, 3, 4])

        rxP = dc.receivers.Pole(locationsM)
        rxD = dc.receivers.Dipole(locationsM, locationsN)

        srcP = dc.sources.Pole([rxP, rxD], locationA)
        srcD = dc.sources.Dipole([rxP, rxD], locationA, locationB)

        survey = dc.Survey([srcP, srcD])

        sim_dict = {'mesh':mesh,
                    'sigmaMap':sigma_map,
                    'survey':survey,
                    'sigma':sigma
                   }

        for s in self.simulations_to_test:
            print(s)
            if '1D' in s:
                sim = getattr(dc, s)(thicknesses=thicks, **sim_dict)
            else:
                sim = getattr(dc, s)(**sim_dict)
            sim2 = dc.simulation.BaseDCSimulation.deserialize(
                sim.serialize(), trusted=True
            )
            self.assertTrue(sim is not sim2)
            self.assertTrue(properties.equal(sim, sim2))

if __name__ == '__main__':
    unittest.main()
