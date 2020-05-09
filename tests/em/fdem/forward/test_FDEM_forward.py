import unittest
import numpy as np
import sys
from scipy.constants import mu_0

from discretize import TensorMesh

from SimPEG import maps
from SimPEG.electromagnetics import frequency_domain as fdem
from SimPEG.electromagnetics import time_domain as tdem
from SimPEG.electromagnetics.utils.testing_utils import getFDEMProblem, crossCheckTest

testEB = True
testHJ = True
testEJ = True
testBH = True
verbose = False

TOLEBHJ = 1e-5
TOLEJHB = 1 # averaging and more sensitive to boundary condition violations (ie. the impact of violating the boundary conditions in each case is different.)
#TODO: choose better testing parameters to lower this

sources_to_test = ["RawVec", "MagDipole_Bfield", "MagDipole", "CircularLoop"]


class SerializationTest(unittest.TestCase):

    receivers_to_test = [
        "PointCurrentDensity",
        "PointElectricField",
        "PointMagneticField",
        "PointMagneticFluxDensity",
        "PointMagneticFluxDensitySecondary",
    ]

    sources_to_test = [
        'CircularLoop',
        'CircularLoopWholeSpace',
        'MagDipole',
        'MagDipole_Bfield',
        'MagneticDipoleWholeSpace',
        # 'PrimSecMappedSigma',   # these are a bit complicated.
        # 'PrimSecSigma',
        'RawVec',
        'RawVec_e',
        'RawVec_m'
    ]

    simulations_to_test = [
        "Simulation3DCurrentDensity",
        "Simulation3DElectricField",
        "Simulation3DMagneticField",
        "Simulation3DMagneticFluxDensity",
    ]

    def test_receiver_serialization(self):
        locations = np.atleast_2d([1, 2, 3])
        orientation = "x"
        component = "real"

        for r in self.receivers_to_test:
            rx = getattr(fdem.receivers, r)(
                locations=locations, orientation=orientation,
                component=component
            )
            rxs = rx.serialize()
            rx2 = fdem.receivers.BaseRx.deserialize(rxs, trusted=True)

            self.assertTrue(rx.storeProjections == rx2.storeProjections)
            self.assertTrue(rx.orientation == rx2.orientation)
            self.assertTrue(rx.component == rx2.component)
            self.assertTrue(np.all(rx.locations == rx2.locations))
            self.assertTrue(rx.__class__ == rx2.__class__)

    def test_source_serialization(self):
        rx = fdem.receivers.PointCurrentDensity(
            locations=np.atleast_2d([1, 2, 3]),
            orientation="x",
            component="real",
        )

        source_parameters = {
            "receiver_list" : [rx],
            "location" : np.r_[0, 0, 0],
            "frequency" : 1.,
            "mu" : 2*mu_0,
            "orientation" : "x",
            "radius" : 2.,
            "current" : 3.,
            "_s_e" : np.random.randn(10),
            "_s_m" : np.random.randn(20),
        }

        for s in self.sources_to_test:
            attrs = {
                key: val for key, val in source_parameters.items() if key in
                dir(getattr(fdem.sources, s))
            }

            src = getattr(fdem.sources, s)(**attrs)
            src2 = fdem.sources.BaseFDEMSrc.deserialize(
                src.serialize(), trusted=True
            )

            for key in attrs.keys():
                if (
                    isinstance(source_parameters[key], np.ndarray) or
                    key == "orientation"
                ):
                    self.assertTrue(np.all(
                        getattr(src, key) == getattr(src2, key)
                    ))
                elif key == "receiver_list":
                    self.assertTrue(
                        len(src.receiver_list) == len(src2.receiver_list)
                    )

                    rx0 = src.receiver_list[0]
                    rx1 = src2.receiver_list[0]
                    self.assertTrue(rx0.__class__ == rx1.__class__)
                    self.assertTrue(np.all(rx0.locations == rx1.locations))
                    self.assertTrue(rx0.orientation == rx1.orientation)
                else:
                    self.assertTrue(
                        getattr(src, key) == getattr(src2, key)
                    )

    def test_survey_serialization(self):
        rx = fdem.receivers.PointCurrentDensity(
            locations=np.atleast_2d([1, 2, 3]),
            orientation="x",
            component="imag",
        )

        src = fdem.sources.MagDipole(
            location=np.r_[0., 3., 0.],
            orientation="z",
            receiver_list=[rx]
        )

        survey = fdem.Survey([src])
        survey2 = fdem.Survey.deserialize(survey.serialize(), trusted=True)

        self.assertTrue(
            np.all(survey.source_list[0].location == survey2.source_list[0].location)
        )
        self.assertTrue(
            np.all(survey.source_list[0].orientation == survey2.source_list[0].orientation)
        )
        self.assertTrue(
            np.all(
                survey.source_list[0].receiver_list[0].locations ==
                survey2.source_list[0].receiver_list[0].locations
            )
        )
        self.assertTrue(
            np.all(
                survey.source_list[0].receiver_list[0].component ==
                survey2.source_list[0].receiver_list[0].component
            )
        )


    def test_simulation_serialization(self):

        mesh = TensorMesh([np.ones(10), np.ones(10), np.ones(10)])
        sigma_map = maps.ExpMap(mesh)

        rx = fdem.receivers.PointCurrentDensity(
            locations=np.atleast_2d([1, 2, 3]),
            orientation="x",
            component="imag",
        )

        src = fdem.sources.MagDipole(
            location=np.r_[0., 3., 0.],
            orientation="z",
            receiver_list=[rx]
        )

        survey = fdem.Survey([src])

        for s in [
            "Simulation3DMagneticField", "Simulation3DElectricField",
            "Simulation3DCurrentDensity", "Simulation3DMagneticFluxDensity"
        ]:
            sim = getattr(fdem, s)(
                mesh=mesh, sigmaMap=sigma_map, survey=survey
            )

            sim2 = fdem.simulation.BaseFDEMSimulation.deserialize(
                sim.serialize(), trusted=True
            )

            self.assertTrue(
                np.all(sim.mesh.gridCC == sim2.mesh.gridCC)
            )

            self.assertTrue(
                isinstance(sim.sigmaMap, type(sim2.sigmaMap))
            )

            self.assertTrue(
                np.all(
                    sim.survey.source_list[0].location ==
                    sim2.survey.source_list[0].location
                )
            )

class SrcLocTest(unittest.TestCase):
    def test_src(self):
        src = fdem.Src.MagDipole(
            [], loc=np.array([[1.5, 3., 5.]]),
            freq=10
        )
        self.assertTrue(np.all(src.location == np.r_[1.5, 3., 5.]))
        self.assertTrue(src.location.shape==(3,))

        with self.assertRaises(Exception):
            src = fdem.Src.MagDipole(
                [], loc=np.array([[0., 0., 0., 1.]]),
                freq=10
            )

        with self.assertRaises(Exception):
            src = fdem.Src.MagDipole(
                [], loc=np.r_[0., 0., 0., 1.],
                freq=10
            )

        src = tdem.Src.MagDipole(
            [], loc=np.array([[1.5, 3., 5.]]),
        )
        self.assertTrue(np.all(src.location == np.r_[1.5, 3., 5.]))

        with self.assertRaises(Exception):
            src = tdem.Src.MagDipole(
                [], loc=np.array([[0., 0., 0., 1.]]),
            )

        with self.assertRaises(Exception):
            src = tdem.Src.MagDipole(
                [], loc=np.r_[0., 0., 0., 1.],
            )

class FDEM_CrossCheck(unittest.TestCase):
    if testEB:
        def test_EB_CrossCheck_exr_Eform(self):
            self.assertTrue(crossCheckTest(sources_to_test, "e", "b", ["ElectricField","x","r"], verbose=verbose))
        def test_EB_CrossCheck_eyr_Eform(self):
            self.assertTrue(crossCheckTest(sources_to_test, "e", "b", ["ElectricField","y","r"], verbose=verbose))
        def test_EB_CrossCheck_ezr_Eform(self):
            self.assertTrue(crossCheckTest(sources_to_test, "e", "b", ["ElectricField","z","r"], verbose=verbose))
        def test_EB_CrossCheck_exi_Eform(self):
            self.assertTrue(crossCheckTest(sources_to_test, "e", "b", ["ElectricField","x","i"], verbose=verbose))
        def test_EB_CrossCheck_eyi_Eform(self):
            self.assertTrue(crossCheckTest(sources_to_test, "e", "b", ["ElectricField","y","i"], verbose=verbose))
        def test_EB_CrossCheck_ezi_Eform(self):
            self.assertTrue(crossCheckTest(sources_to_test, "e", "b", ["ElectricField","z","i"], verbose=verbose))

        def test_EB_CrossCheck_bxr_Eform(self):
            self.assertTrue(crossCheckTest(sources_to_test, "e", "b", ["MagneticFluxDensity","x","r"], verbose=verbose))
        def test_EB_CrossCheck_byr_Eform(self):
            self.assertTrue(crossCheckTest(sources_to_test, "e", "b", ["MagneticFluxDensity","y","r"], verbose=verbose))
        def test_EB_CrossCheck_bzr_Eform(self):
            self.assertTrue(crossCheckTest(sources_to_test, "e", "b", ["MagneticFluxDensity","z","r"], verbose=verbose))
        def test_EB_CrossCheck_bxi_Eform(self):
            self.assertTrue(crossCheckTest(sources_to_test, "e", "b", ["MagneticFluxDensity","x","i"], verbose=verbose))
        def test_EB_CrossCheck_byi_Eform(self):
            self.assertTrue(crossCheckTest(sources_to_test, "e", "b", ["MagneticFluxDensity","y","i"], verbose=verbose))
        def test_EB_CrossCheck_bzi_Eform(self):
            self.assertTrue(crossCheckTest(sources_to_test, "e", "b", ["MagneticFluxDensity","z","i"], verbose=verbose))

    if testHJ:
        def test_HJ_CrossCheck_jxr_Jform(self):
            self.assertTrue(crossCheckTest(sources_to_test, "j", "h", ["CurrentDensity","x","r"], verbose=verbose))
        def test_HJ_CrossCheck_jyr_Jform(self):
            self.assertTrue(crossCheckTest(sources_to_test, "j", "h", ["CurrentDensity","y","r"], verbose=verbose))
        def test_HJ_CrossCheck_jzr_Jform(self):
            self.assertTrue(crossCheckTest(sources_to_test, "j", "h", ["CurrentDensity","z","r"], verbose=verbose))
        def test_HJ_CrossCheck_jxi_Jform(self):
            self.assertTrue(crossCheckTest(sources_to_test, "j", "h", ["CurrentDensity","x","i"], verbose=verbose))
        def test_HJ_CrossCheck_jyi_Jform(self):
            self.assertTrue(crossCheckTest(sources_to_test, "j", "h", ["CurrentDensity","y","i"], verbose=verbose))
        def test_HJ_CrossCheck_jzi_Jform(self):
            self.assertTrue(crossCheckTest(sources_to_test, "j", "h", ["CurrentDensity","z","i"], verbose=verbose))

        def test_HJ_CrossCheck_hxr_Jform(self):
            self.assertTrue(crossCheckTest(sources_to_test, "j", "h", ["MagneticField","x","r"], verbose=verbose))
        def test_HJ_CrossCheck_hyr_Jform(self):
            self.assertTrue(crossCheckTest(sources_to_test, "j", "h", ["MagneticField","y","r"], verbose=verbose))
        def test_HJ_CrossCheck_hzr_Jform(self):
            self.assertTrue(crossCheckTest(sources_to_test, "j", "h", ["MagneticField","z","r"], verbose=verbose))
        def test_HJ_CrossCheck_hxi_Jform(self):
            self.assertTrue(crossCheckTest(sources_to_test, "j", "h", ["MagneticField","x","i"], verbose=verbose))
        def test_HJ_CrossCheck_hyi_Jform(self):
            self.assertTrue(crossCheckTest(sources_to_test, "j", "h", ["MagneticField","y","i"], verbose=verbose))
        def test_HJ_CrossCheck_hzi_Jform(self):
            self.assertTrue(crossCheckTest(sources_to_test, "j", "h", ["MagneticField","z","i"], verbose=verbose))

if __name__ == "__main__":
    unittest.main()
