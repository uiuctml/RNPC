import enum

class LogLevel(enum.Enum):
    all = 6
    trace = 5
    debug = 4
    info = 3
    warn = 2
    error = 1
    fatal = 0
    off = -1

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

class ModelBaseline(enum.Enum):
    resnet152 = 0
    vit_b_32 = 1

class ModelDecomposed(enum.Enum):
    cnn_mtl = 0
    cnn_set = 1
    mlp_set = 2
    mlp_set_3 = 3
    resnet152_mtl = 4
    vit_b_32_mtl = 5

class ModelReference(enum.Enum):
    cbm = 0
    cbm_cat = 1
    cem = 2
    dcr = 3
    len = 4

class OptimizerSPN(enum.Enum):
    cccp_discriminative = 0
    cccp_generative = 1
    ebw_discriminative = 2
    pgd_discriminative = 3
    pgd_generative = 4
