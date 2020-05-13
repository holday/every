import numpy as np
import properties

from .... import survey
from ....utils import Zero, closestPoints
from ....utils.code_utils import deprecate_property
from .receivers import BaseRx


class BaseSrc(survey.BaseSrc):
    """
    Base DC source
    """
    _REGISTRY = {}

    current = properties.Float("amplitude of the source current", default=1.)

    receiver_list = properties.List(
        "receiver list",
        properties.Instance("a SimPEG.electromagnetics.static.resistivity receiver", BaseRx),
        default=[]
    )

    _q = None

    def eval(self, prob):
        raise NotImplementedError

    def evalDeriv(self, prob):
        return Zero()


class Dipole(BaseSrc):
    """
    Dipole source
    """

    location = properties.List(
        "location of the source electrodes",
        survey.SourceLocationArray("location of electrode")
    )
    loc = deprecate_property(location, 'loc', removal_version='0.15.0')

    def __init__(self, receiver_list=None, locationA=None, locationB=None, **kwargs):
        loc = kwargs.pop('location', None)
        if loc is None:
            if locationA.shape != locationB.shape:
                raise Exception('Shape of locationA and locationB should be the same')
            self.location = [locationA, locationB]
        else:
            self.location = loc
        super().__init__(receiver_list=receiver_list, **kwargs)

    def eval(self, prob):
        if self._q is not None:
            return self._q
        else:
            if prob._formulation == 'HJ':
                inds = closestPoints(prob.mesh, self.location, gridLoc='CC')
                self._q = np.zeros(prob.mesh.nC)
                self._q[inds] = self.current * np.r_[1., -1.]
            elif prob._formulation == 'EB':
                qa = prob.mesh.getInterpolationMat(
                    self.location[0], locType='N'
                ).toarray()
                qb = -prob.mesh.getInterpolationMat(
                    self.location[1], locType='N'
                ).toarray()
                self._q = self.current * (qa+qb)
            return self._q


class Pole(BaseSrc):

    def eval(self, prob):
        if self._q is not None:
            return self._q
        else:
            if prob._formulation == 'HJ':
                inds = closestPoints(prob.mesh, self.location)
                self._q = np.zeros(prob.mesh.nC)
                self._q[inds] = self.current * np.r_[1.]
            elif prob._formulation == 'EB':
                q = prob.mesh.getInterpolationMat(
                    self.location, locType='N'
                )
                self._q = self.current * q.toarray()
            return self._q
