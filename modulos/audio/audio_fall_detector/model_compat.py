"""Carga del joblib original NumPy 2 en el entorno visual NumPy 1.26.

Adaptación local de nombres/constructores de RNG durante deserialización.
No modifica NumPy globalmente ni altera arrays, pesos o clases del modelo.
El artefacto original es joblib sin compresión (protocolo pickle 4).
"""
import numpy as np
from joblib.numpy_pickle import NumpyUnpickler


class _CompatibleMT19937(np.random.MT19937):
    def __setstate__(self, state):
        # NumPy 2 añade SeedSequence como segundo elemento de la tupla.
        state = dict(state[0] if isinstance(state, tuple) else state)
        state['bit_generator'] = type(self).__name__
        self.state = state


def _bit_generator(value='MT19937'):
    if value == 'MT19937' or value is np.random.MT19937:
        return _CompatibleMT19937()
    if isinstance(value, str):
        value = getattr(np.random, value)
    return value() if isinstance(value, type) else value


def _random_state(value='MT19937', constructor=_bit_generator):
    generator = constructor(value)
    if isinstance(generator, _CompatibleMT19937):
        native = np.random.MT19937()
        state = generator.state
        state['bit_generator'] = 'MT19937'
        native.state = state
        generator = native
    return np.random.RandomState(generator)


class _CompatibleUnpickler(NumpyUnpickler):
    def find_class(self, module, name):
        if module == 'numpy.random._pickle':
            if name == '__bit_generator_ctor':
                return _bit_generator
            if name == '__randomstate_ctor':
                return _random_state
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)


def load_original_model(path):
    import warnings
    with open(path, 'rb') as handle, warnings.catch_warnings():
        # El artefacto se guardó con scikit-learn 1.9.1; con Python 3.10 se usa
        # 1.7.2. tests/test_audio_model.py comprueba que pesos y predicciones
        # coinciden con la referencia, así que el aviso de versión se omite.
        try:
            from sklearn.exceptions import InconsistentVersionWarning
            warnings.simplefilter('ignore', InconsistentVersionWarning)
        except ImportError:
            pass
        return _CompatibleUnpickler(str(path), handle, ensure_native_byte_order=True).load()
