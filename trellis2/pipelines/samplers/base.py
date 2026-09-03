# Copied from TRELLIS.2 (https://github.com/microsoft/TRELLIS.2/blob/main/trellis2/pipelines/samplers/base.py)
# Copyright (c) Microsoft Corporation. Licensed under the MIT License.

from typing import *
from abc import ABC, abstractmethod


class Sampler(ABC):
    """
    A base class for samplers.
    """

    @abstractmethod
    def sample(
        self,
        model,
        **kwargs
    ):
        """
        Sample from a model.
        """
        pass
    